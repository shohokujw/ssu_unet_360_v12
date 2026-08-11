"""Reconstruct from network-synthesised views instead of the 9 measured ones.

This is the v12 replacement for `step1_make_fbp.py`. Everything downstream
(step4 dataset, step5 training) is unchanged in shape -- it still receives one
`{'fbp_lh': volume}` pickle per subject -- only the reconstruction that goes in
is denser.

    step1_make_fbp.py    9 measured views                  -> fbp_lh
    step1b (this)        9 measured + N synthesised views   -> fbp_interp

The views come from the delta-conditioned interpolator in
`sinogram_interpolation_pix2pix_v9`, applied by recursive bisection: every
angular gap wider than `--target_gap` gets a predicted midpoint, repeatedly,
until none is left. A 20 deg acquisition reaches a 5 deg grid in two rounds
(9 -> 17 -> 33); the sequences with 40 deg gaps need three (-> 41).

Why this is worth doing: with 9 views the reconstruction error is dominated by
long streaks that span the whole image, and a CNN denoiser is poor at removing
those. With the synthesised views the error becomes short local texture, which
is what a denoiser handles well -- so the gain downstream may exceed the ~2.5 dB
the reconstruction itself picks up.

IMPORTANT -- the interpolator must not have trained on this project's test
subjects. Rebuild its cache with
`Step1_make_cache.py --split_from <this params.yml>` so the two projects share
one split; this script verifies that and refuses to run otherwise.

Usage
-----
    python step1b_make_fbp_interp.py                       # all subjects
    python step1b_make_fbp_interp.py --target_gap 10       # coarser
    python step1b_make_fbp_interp.py --subjects 0004 0017  # a few
"""

import argparse
import json
import os
import pickle
import sys
import time
from glob import glob

import numpy as np
import torch
import yaml

current_folder = os.path.dirname(os.path.realpath(__file__))

from op_ct_sstlabs import FBP, Set_operation, get_params  # noqa: E402
from preprocessing import get_obj_height, apply_corner_triangles_3d  # noqa: E402
from config_utils import load_ct_params  # noqa: E402

X, Y, Z = 0, 1, 2


# ----------------------------------------------------------------------------
# Interpolator (lives in the sinogram-interpolation project)
# ----------------------------------------------------------------------------

def load_interpolator(project, ckpt_path, device):
    """Return (netG, use_abs_angle, trained_gaps, helper fns) from that project.

    Only the model and two pure-python helpers are imported; the project's own
    CT operators are deliberately left alone so this script uses v12's.
    """
    sys.path.insert(0, project)
    sys.path.insert(0, os.path.join(project, 'include'))
    sys.path.insert(0, os.path.join(project, 'Pix2Pix'))

    from model_v10 import CondUNet
    from dataloader_v10 import make_cond, cond_dim
    from geometry import resolve_angles

    cfg_path = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)),
                            'experiment_config.json')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f'{cfg_path} not found; it records the architecture the checkpoint '
            'was trained with and there is no safe default to guess.')
    with open(cfg_path, 'r') as f:
        train_args = json.load(f)['args']

    use_abs_angle = not train_args.get('no_abs_angle', False)
    netG = CondUNet(nch_in=2, nch_out=1,
                    nch_ker=train_args['nch_ker'], norm=train_args['norm'],
                    num_down=train_args['num_down_G'], dropout=train_args['dropout'],
                    cond_dim=cond_dim(use_abs_angle), emb_dim=train_args['emb_dim'])
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    netG.load_state_dict(ckpt['netG'])
    netG.to(device).eval()

    print(f'Interpolator: {ckpt_path}')
    print(f'  epoch {ckpt["epoch"]}, trained gaps {train_args["gaps"]}, '
          f'model_type {train_args["model_type"]}, num_down_G {train_args["num_down_G"]}')

    return netG, use_abs_angle, [float(g) for g in train_args['gaps']], \
        make_cond, resolve_angles, train_args


def verify_no_leak(ckpt_path, params):
    """Refuse to run if the interpolator trained on our val/test subjects.

    The split is read from the checkpoint's own experiment_config.json, NOT from
    the interpolation project's cache manifest. The manifest records whatever
    split was last written and can be rebuilt after training, so checking
    against it can pass for the wrong reason -- exactly the false negative this
    function exists to prevent.
    """
    held_out = set(params['dataset_split']['val']) | set(params['dataset_split']['test'])

    cfg_path = os.path.join(os.path.dirname(os.path.dirname(ckpt_path)),
                            'experiment_config.json')
    with open(cfg_path, 'r') as f:
        cfg = json.load(f)

    if 'split' not in cfg:
        print()
        print('!' * 76)
        print('CANNOT VERIFY THE SPLIT')
        print(f'  {cfg_path}')
        print('  has no "split" entry, so which subjects this interpolator saw is')
        print('  unknown. Its synthesised views for a subject it trained on may be')
        print('  better than deployment would give, which inflates any downstream')
        print('  result measured on that subject.')
        print('  Checkpoints trained after this check was added record their split.')
        print('!' * 76)
        print()
        return

    interp_train = set(cfg['split']['train'])
    leak = sorted(held_out & interp_train)
    if leak:
        raise RuntimeError(
            f'{len(leak)} of our held-out subjects are in the interpolator\'s '
            f'TRAINING set: {leak}\n'
            'Their synthesised views would come from a network that has already '
            'seen them, so any downstream result on them is optimistic.\n'
            'Fix: rebuild that cache with\n'
            f'  python Step1_make_cache.py --split_from {current_folder}/include/params.yml\n'
            'and retrain the interpolator.')
    print(f'Split check OK: none of our {len(held_out)} val/test subjects appear '
          f"in the interpolator's training split.")


# ----------------------------------------------------------------------------
# Recursive bisection
# ----------------------------------------------------------------------------

def synthesize(views, angles, netG, device, make_cond, use_abs_angle,
               target_gap, gap_tol, vmin, vmax, batch_size, max_levels=6):
    """Insert predicted midpoints until every gap is at most `target_gap`.

    Args:
        views: ``(n, vd, ud)`` float32 in raw projection units, ascending angle.
        angles: ``(n,)`` degrees, ascending and unwrapped (so a -10 deg source
            stays next to 30 rather than sorting to 350 and faking a 160 deg gap).

    Returns:
        (views, angles) with the synthesised views merged in, still sorted.
    """
    def norm(x):
        return (np.clip(x, vmin, vmax).astype(np.float32) - vmin) / (vmax - vmin) * 2.0 - 1.0

    def denorm(x):
        return (x + 1.0) * 0.5 * (vmax - vmin) + vmin

    for level in range(1, max_levels + 1):
        pairs = [i for i in range(len(angles) - 1)
                 if (angles[i + 1] - angles[i]) > target_gap + gap_tol]
        if not pairs:
            break

        new_views, new_angles = [], []
        for s in range(0, len(pairs), batch_size):
            chunk = pairs[s:s + batch_size]
            x = np.stack([np.stack([views[i], views[i + 1]], axis=0) for i in chunk])
            x = torch.from_numpy(norm(x)).to(device)

            conds, mids = [], []
            for i in chunk:
                gap = float(angles[i + 1] - angles[i])
                mids.append(float(angles[i] + gap / 2.0))
                conds.append(make_cond(gap, 0.5, float(angles[i]), use_abs_angle))
            c = torch.stack(conds).to(device)

            with torch.no_grad():
                out = netG(x, c)
            new_views.append(denorm(out[:, 0].cpu().numpy().astype(np.float32)))
            new_angles.extend(mids)

        views = np.concatenate([views, np.concatenate(new_views, axis=0)], axis=0)
        angles = np.concatenate([angles, np.asarray(new_angles, dtype=np.float64)])
        order = np.argsort(angles)
        views, angles = views[order], angles[order]
        print(f'      level {level}: +{len(pairs)} -> {len(angles)} views, '
              f'max gap {np.diff(angles).max():.1f}°')

    return views, angles


# ----------------------------------------------------------------------------
# FBP from an explicit angle list
# ----------------------------------------------------------------------------

def fbp_from_views(prj, angles_deg, obj_height, base_params, so_dir, img_size, nz):
    """FBP of a view stack whose angles have no calibration entry of their own.

    Args:
        prj: ``(nz, n_views, ud)`` float32, view order matching `angles_deg`.
        angles_deg: ascending, unwrapped, in `ang_primary` convention.
    """
    n = len(angles_deg)
    p = dict(base_params)
    ang = np.asarray(angles_deg, dtype=np.float32)
    zeros = np.zeros(n, dtype=np.float32)

    p['ang_primary'] = ang
    p['angle_offset'] = zeros.copy()
    p['det_tilt'] = zeros.copy()
    p['ang_iso_center'] = ((ang - 90.0) * np.pi / 180.0).astype(np.float32)
    p['SID'] = np.full(n, base_params['full_SID'], np.float32)
    p['SOD'] = np.full(n, base_params['full_SOD'], np.float32)
    p['sparse_view'] = n

    p['img_mat'] = np.array([img_size, img_size, nz], dtype=np.int32)
    voxel_xy = 512.0 / img_size
    p['voxel'] = np.array([voxel_xy, voxel_xy, 1.0], dtype=np.float32)
    p['sample'] = float(min(p['voxel'][X], p['voxel'][Y]) / 2.0)

    op = Set_operation(so_file_dir=so_dir, **p)

    # FBP mutates ang_iso_center in place -> hand it a private copy.
    p_call = dict(p)
    p_call['ang_iso_center'] = p['ang_iso_center'].copy()
    return FBP(prj=np.ascontiguousarray(prj, dtype=np.float32),
               obj_height=obj_height, mode='sparse', OP_CT=op, **p_call)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='FBP from network-synthesised views (v12 replacement for step1)')
    ap.add_argument('--target_gap', type=float, default=5.0,
                    help='Bisect until every angular gap is at most this (deg)')
    ap.add_argument('--gap_tol', type=float, default=0.5,
                    help='Slack on --target_gap; the acquisition grid is '
                         '~0.997 deg/view so exact halves are unreachable')
    ap.add_argument('--img_size', type=int, default=512)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--vmin', type=float, default=-2.0)
    ap.add_argument('--vmax', type=float, default=5.0)
    ap.add_argument('--out_name', type=str, default='fbp_interp')
    ap.add_argument('--subjects', type=str, nargs='*', default=None,
                    help='Subject ids to process (default: all)')
    ap.add_argument('--gpu_id', type=int, default=0)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--skip_leak_check', action='store_true',
                    help='Only for debugging; results become uninterpretable')
    args = ap.parse_args()

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    yaml_path = f'{current_folder}/include/params.yml'
    cfg = load_ct_params()

    model_name = cfg['model_name']
    ang_list = sorted(cfg['ang'])
    w_l, w_h = cfg['w_l'], cfg['w_h']
    prj_root = cfg['prj_path']
    so_dir = f'{current_folder}/include'

    project = cfg['interp_project']
    ckpt = cfg['interp_ckpt']

    print(f'model_name : {model_name}')
    print(f'angles     : {ang_list}')
    print(f'target gap : {args.target_gap}°  (+{args.gap_tol}° slack)')

    if not args.skip_leak_check:
        verify_no_leak(ckpt, cfg)

    netG, use_abs_angle, trained_gaps, make_cond, resolve_angles, _ = \
        load_interpolator(project, ckpt, device)

    dir_save = f'{current_folder}/datasets/{model_name}/{args.out_name}'
    os.makedirs(dir_save, exist_ok=True)
    print(f'output     : {dir_save}\n')

    lst = sorted(glob(os.path.join(prj_root, '*.pkl')))
    th = 18 if args.img_size == 512 else 9
    ud = cfg['Ud']

    for path_data in lst:
        sub = os.path.basename(path_data).split('_')[-1][:-4]
        if args.subjects and sub not in args.subjects:
            continue
        path_save = os.path.join(dir_save, f'FBP_{sub}.pkl')
        if os.path.exists(path_save) and not args.overwrite:
            print(f'[{sub}] exists, skipping')
            continue

        print(f'[{sub}]')
        with open(path_data, 'rb') as f:
            d = pickle.load(f)
        prj_l_all, prj_h_all = d['prj_l'], d['prj_h']
        del d

        # Per-subject calibrated geometry, as in step1_make_fbp.py.
        geo_path = f'{current_folder}/geo_param/{sub}.yaml'
        params = get_params(file_path=yaml_path, geo_path=geo_path, view_9=True)
        with open(geo_path, 'r') as f:
            view_angles = np.asarray(yaml.safe_load(f)['ang_primary'], dtype=np.float64)

        # Angles resolved on this subject's grid and unwrapped, so consecutive
        # differences are the real acquisition gaps.
        idx9, ang9 = resolve_angles(view_angles, ang_list)

        prj_l = prj_l_all[:, idx9, :].copy()
        nz = len(prj_l)

        # obj_height from the low-energy 9-view stack, identical to step1.
        params['img_mat'][2] = nz
        obj_height = get_obj_height(prj_l, 1, ud, args.img_size, params)

        # The interpolator was trained on the weighted combination, so combine
        # first and synthesise once -- not once per energy.
        prj_lh = (w_l * prj_l + w_h * prj_h_all[:, idx9, :]).astype(np.float32)
        del prj_l, prj_l_all, prj_h_all

        views0 = np.ascontiguousarray(prj_lh.transpose(1, 0, 2))   # (9, vd, ud)
        gaps = sorted({round(float(g), 1) for g in np.diff(ang9)})
        untrained = [g for g in gaps
                     if not any(abs(g - t) < 2.0 for t in trained_gaps)]
        if untrained:
            print(f'      WARNING: level-1 gaps {untrained}° are outside the '
                  f'trained set {trained_gaps}°; those are extrapolation.')

        t0 = time.monotonic()
        views, angles = synthesize(views0, ang9, netG, device, make_cond,
                                   use_abs_angle, args.target_gap, args.gap_tol,
                                   args.vmin, args.vmax, args.batch_size)

        prj_syn = np.ascontiguousarray(views.transpose(1, 0, 2))    # (nz, N, ud)
        fbp_lh = fbp_from_views(prj_syn, angles, obj_height, params, so_dir,
                                args.img_size, nz)
        print(f'      synth + FBP: {time.monotonic() - t0:.1f}s')

        fbp_lh[fbp_lh < 0] = 0
        fbp_lh = apply_corner_triangles_3d(fbp_lh, th)
        print(f'      {len(angles)} views -> {fbp_lh.shape}, '
              f'range [{fbp_lh.min():.4f}, {fbp_lh.max():.4f}]')

        with open(path_save, 'wb') as f:
            pickle.dump({'fbp_lh': fbp_lh,
                         'n_views': int(len(angles)),
                         'angles': [float(a) for a in angles]}, f)
        del views, prj_syn, fbp_lh

    print('\nDone. Next: recalibrate the normalisation constants with '
          'step1c_calibrate_norm.py -- the ones in params.yml were fitted to '
          '9-view FBP and this distribution is different.')


if __name__ == '__main__':
    main()
