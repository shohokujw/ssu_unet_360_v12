"""
Evaluation Script (Step 7)

step6 writes reconstructions but no numbers. This measures them against the
720-view label and, in the same table, measures the network's *input* against
the same label -- so the row that matters is not "how good is the output" but
"how much did the denoiser actually add over the reconstruction it was given".

Everything is compared in physical units, inverted with the constant step4
normalised the label with (see config_utils.get_label_max_lh). Inverting with
Pix2Pix/model_v2.normalize_param() instead scales values by 1.375x and makes
every number here meaningless.

Metrics, per slice, averaged over the volume:
    MAE   mean |output - label|
    RMSE  sqrt(mean (output - label)^2)
    PSNR  20*log10(peak / RMSE), peak = the label's own max (see --peak)
    SSIM  structural similarity, if scikit-image is installed

Usage:
    python3 step7_Evaluate.py                          # test split, best ckpt run
    python3 step7_Evaluate.py --sets test val
    python3 step7_Evaluate.py --csv metrics.csv        # also write per-subject rows
"""

import argparse
import os
import pickle
import sys

import numpy as np

current_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, current_folder)

from config_utils import load_ct_params, get_label_max_lh
from preprocessing import denormalize

try:
    from skimage.metrics import structural_similarity as _ssim
except ImportError:
    _ssim = None


def slice_metrics(pred, ref, peak):
    """MAE, RMSE, PSNR, SSIM for one slice, all in physical units."""
    diff = pred - ref
    mae = float(np.abs(diff).mean())
    rmse = float(np.sqrt((diff ** 2).mean()))
    psnr = float('inf') if rmse == 0 else float(20 * np.log10(peak / rmse))
    ssim = (float(_ssim(ref, pred, data_range=peak)) if _ssim is not None
            else float('nan'))
    return mae, rmse, psnr, ssim


def load_label_volume(fin_dir, subject, z_slice, label_name, max_lh):
    """Per-slice label pickles -> one physical-unit volume."""
    out = []
    for z in range(z_slice):
        path = os.path.join(fin_dir, subject, f'{label_name}_{subject}_{z:03d}.pkl')
        if not os.path.exists(path):
            break
        with open(path, 'rb') as f:
            out.append(pickle.load(f)['fbp_lh'])
    if not out:
        return None
    return denormalize(np.stack(out), max_value=max_lh)


def load_input_volume(fin_dir, subject, z_slice, max_lh_in):
    """The same slices the network was fed, back in physical units."""
    out = []
    for z in range(z_slice):
        path = os.path.join(fin_dir, subject, f'FBP_{subject}_{z:03d}.pkl')
        if not os.path.exists(path):
            break
        with open(path, 'rb') as f:
            out.append(pickle.load(f)['fbp_lh'])
    if not out:
        return None
    return denormalize(np.stack(out), max_value=max_lh_in)


def evaluate(args):
    params = load_ct_params()
    img_size = args.img_size
    data_name = params['model_name']
    z_slice = params['training']['z_slices']
    label_name = params['label_name']
    input_name = params.get('input_name', 'fbp_lh')

    max_lh = get_label_max_lh(params, img_size)
    n_in = params['normalization'].get(
        input_name, params['normalization']['fbp'])[img_size]
    max_lh_in = params['w_l'] * n_in['max_l'] + params['w_h'] * n_in['max_h']

    exp = (f"ndG{args.num_downs_G}_ndD{args.num_downs_D}_dropout_3"
           + ('' if input_name == 'fbp_lh' else f'__{input_name}'))
    result_root = args.pred_dir or os.path.join(
        current_folder, args.model_name, data_name, exp, 'result', str(img_size))
    fin_dir_root = args.fin_dir or os.path.join(
        current_folder, 'datasets', data_name, 'fin_set', str(img_size))

    # The baseline's input is a different reconstruction with its own constant.
    if args.input_norm:
        n_in = params['normalization'][args.input_norm][img_size]
        max_lh_in = params['w_l'] * n_in['max_l'] + params['w_h'] * n_in['max_h']
        input_name = args.input_norm

    # Predictions produced by code that inverted with a different constant can
    # still be scored: the error is a pure scale factor.
    if args.pred_max_lh:
        print(f'rescaling predictions by {max_lh / args.pred_max_lh:.6f} '
              f'({args.pred_max_lh} -> {max_lh})')

    print(f'experiment   : {args.label or exp}')
    print(f'label max_lh : {max_lh:.4f}   input ({input_name}) max_lh: {max_lh_in:.4f}')
    if _ssim is None:
        print('NOTE: scikit-image not installed -- SSIM will be nan')
    print()

    rows = []
    for set_name in args.sets:
        pred_dir = os.path.join(result_root, set_name)
        fin_dir = os.path.join(fin_dir_root, set_name)
        if not os.path.isdir(pred_dir):
            print(f'[{set_name}] no reconstructions in {pred_dir} -- run step6 first')
            continue

        subjects = sorted(d for d in os.listdir(fin_dir)
                          if os.path.isdir(os.path.join(fin_dir, d)))
        print(f'[{set_name}] {len(subjects)} subjects')
        print(f'{"subject":>8}  {"MAE":>10} {"RMSE":>10} {"PSNR":>7} {"SSIM":>6}   '
              f'{"inMAE":>10} {"inPSNR":>7} {"inSSIM":>6}')

        for sub in subjects:
            pred_path = os.path.join(pred_dir, f'FIN_{sub}.pkl')
            if not os.path.exists(pred_path):
                print(f'{sub:>8}  (no FIN_{sub}.pkl, skipped)')
                continue
            with open(pred_path, 'rb') as f:
                pred = np.asarray(pickle.load(f)['fin_lh'], dtype=np.float32)
            if args.pred_max_lh:
                pred = pred * (max_lh / args.pred_max_lh)

            ref = load_label_volume(fin_dir, sub, z_slice, label_name, max_lh)
            if ref is None:
                print(f'{sub:>8}  (no label slices, skipped)')
                continue
            inp = load_input_volume(fin_dir, sub, z_slice, max_lh_in)

            n = min(len(pred), len(ref))
            peak = args.peak if args.peak else float(ref[:n].max())
            if peak <= 0:
                print(f'{sub:>8}  (empty label, skipped)')
                continue

            m = np.mean([slice_metrics(pred[z], ref[z], peak) for z in range(n)],
                        axis=0)
            if inp is not None:
                mi = np.mean([slice_metrics(inp[z], ref[z], peak)
                              for z in range(min(n, len(inp)))], axis=0)
            else:
                mi = [float('nan')] * 4

            rows.append((set_name, sub, *m, *mi))
            print(f'{sub:>8}  {m[0]:10.6f} {m[1]:10.6f} {m[2]:7.2f} {m[3]:6.4f}   '
                  f'{mi[0]:10.6f} {mi[2]:7.2f} {mi[3]:6.4f}')

        got = [r for r in rows if r[0] == set_name]
        if got:
            a = np.mean([r[2:] for r in got], axis=0)
            print(f'{"MEAN":>8}  {a[0]:10.6f} {a[1]:10.6f} {a[2]:7.2f} {a[3]:6.4f}   '
                  f'{a[4]:10.6f} {a[6]:7.2f} {a[7]:6.4f}')
            print(f'{"":>8}  network vs its input: MAE {a[4] / a[0]:.2f}x lower, '
                  f'PSNR +{a[2] - a[6]:.2f} dB, SSIM +{a[3] - a[7]:.4f}')
        print()

    if args.csv and rows:
        import csv
        with open(args.csv, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['set', 'subject', 'mae', 'rmse', 'psnr', 'ssim',
                        'in_mae', 'in_rmse', 'in_psnr', 'in_ssim'])
            w.writerows(rows)
        print(f'wrote {args.csv}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--img_size', type=int, default=512)
    ap.add_argument('--num_downs_G', type=int, default=7)
    ap.add_argument('--num_downs_D', type=int, default=4)
    ap.add_argument('--model_name', type=str, default='Pix2Pix')
    ap.add_argument('--sets', nargs='+', default=['test'],
                    choices=['train', 'test', 'val'])
    ap.add_argument('--peak', type=float, default=None,
                    help='PSNR peak. Default: each subject\'s own label max, '
                         'which keeps PSNR comparable across methods on the '
                         'same subject but not across subjects.')
    ap.add_argument('--pred_dir', type=str, default=None,
                    help='result/<img_size> directory holding FIN_*.pkl, if not '
                         'this experiment\'s.')
    ap.add_argument('--fin_dir', type=str, default=None,
                    help='fin_set/<img_size> directory holding the labels and '
                         'the inputs those predictions were made from.')
    ap.add_argument('--input_norm', type=str, default=None,
                    help='normalization key for that input (e.g. "fbp" for the '
                         'v11 9-view baseline). Default: this project\'s '
                         'input_name.')
    ap.add_argument('--pred_max_lh', type=float, default=None,
                    help='Constant the predictions were denormalised with, if '
                         'it was not the label\'s. Values are rescaled before '
                         'scoring.')
    ap.add_argument('--label', type=str, default=None,
                    help='Name for this run in the output.')
    ap.add_argument('--csv', type=str, default=None)
    evaluate(ap.parse_args())
