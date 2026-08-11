"""Recalibrate the input normalisation constant for a new reconstruction.

Why this exists
---------------
`preprocessing.normalize` maps a volume with

    out = 2*x/max_value - 1          -> [-1, 1] when x in [0, max_value]

so `max_value` sets the range the network sees. The constants in `params.yml`
under `normalization.fbp` were fitted to 9-VIEW FBP. A reconstruction from
synthesised views is a different distribution -- the streaks that produced the
bright tail are largely gone, and the FBP operator's `dbeta` weighting changes
with view count -- so reusing them mis-scales the input, silently.

The subtle part: matching the BASIS, not just remeasuring
---------------------------------------------------------
It is not enough to compute some statistic of the new data. The existing
constant embodies a particular relationship to its own data:

    v11 fbp_lh:  global max 0.385, p99.9 0.142, constant 0.51
                 -> 1.32x the global max, 3.59x the p99.9

Setting the new constant to its own p99.9 would put the new input roughly 4x
hotter *relative to its own distribution* than the old one was, and the A/B
comparison would then be measuring the normalisation change as much as the
input change. So this script derives the headroom factor from the reference
input and its existing constant, then applies that same factor to the new one.

Both bases are reported. They should agree; if they diverge the two
distributions differ in shape, not just scale, and the choice needs thought.

Usage
-----
    python step1c_calibrate_norm.py
    python step1c_calibrate_norm.py --input_name fbp_interp \
        --ref_dir /home/jwlee/ssu_unet_360_v11/datasets/model_Gen3_7/fbp_lh
"""

import argparse
import os
import pickle
from glob import glob

import numpy as np

from config_utils import load_ct_params, get_dataset_split
from file_utils import extract_subject_id

current_folder = os.path.dirname(os.path.realpath(__file__))


def measure(files, percentile):
    """(global max, mean per-subject percentile) over `files`."""
    mx, pc = [], []
    for f in files:
        with open(f, 'rb') as fid:
            vol = np.asarray(pickle.load(fid)['fbp_lh'], dtype=np.float32)
        nz = vol[vol > 0]                 # background is exactly 0 after clipping
        mx.append(float(vol.max()))
        pc.append(float(np.percentile(nz, percentile)) if nz.size else 0.0)
        del vol, nz
    return float(np.max(mx)), float(np.mean(pc))


def main():
    ap = argparse.ArgumentParser(description='Fit the input normalisation constant')
    ap.add_argument('--input_name', type=str, default=None,
                    help='Directory under datasets/<model>/ to calibrate '
                         '(default: params.yml input_name)')
    ap.add_argument('--ref_dir', type=str,
                    default='/home/jwlee/ssu_unet_360_v11/datasets/model_Gen3_7/fbp_lh',
                    help='Reconstruction the EXISTING constant was fitted to. Its '
                         'ratio to that data is the headroom this script preserves.')
    ap.add_argument('--ref_key', type=str, default='fbp',
                    help='normalization.<key> holding the existing constant')
    ap.add_argument('--percentile', type=float, default=99.9)
    ap.add_argument('--max_subjects', type=int, default=0, help='0 = all train subjects')
    args = ap.parse_args()

    cfg = load_ct_params()
    model_name = cfg['model_name']
    input_name = args.input_name or cfg.get('input_name', 'fbp_interp')
    img_size = cfg['img_mat'][0]

    train_subs = set(get_dataset_split(cfg)[0])

    def train_files(src):
        fs = [f for f in sorted(glob(os.path.join(src, '*.pkl')))
              if extract_subject_id(f) in train_subs]
        return fs[:args.max_subjects] if args.max_subjects else fs

    new_dir = f'{current_folder}/datasets/{model_name}/{input_name}'
    new_files = train_files(new_dir)
    if not new_files:
        raise FileNotFoundError(f'No train-split volumes in {new_dir}. Run step1b first.')

    ref_files = train_files(args.ref_dir)
    ref_const = cfg['normalization'][args.ref_key][img_size]['max_h']

    print(f'new  : {input_name}  ({len(new_files)} train subjects)')
    print(f'ref  : {args.ref_dir}  ({len(ref_files)} train subjects)')
    print(f'       existing normalization.{args.ref_key}.{img_size}.max_h = {ref_const}\n')

    n_max, n_pct = measure(new_files, args.percentile)
    print(f'new  global max {n_max:.4f}   mean p{args.percentile} {n_pct:.4f}')

    if not ref_files:
        print('\nWARNING: reference directory empty; falling back to the raw '
              'percentile, which does NOT preserve the existing headroom.')
        rec = n_pct
        by_max = by_pct = None
    else:
        r_max, r_pct = measure(ref_files, args.percentile)
        print(f'ref  global max {r_max:.4f}   mean p{args.percentile} {r_pct:.4f}')
        print(f'\nheadroom in the existing constant:'
              f'  {ref_const / r_max:.2f}x global max,'
              f'  {ref_const / r_pct:.2f}x p{args.percentile}')
        by_max = n_max * (ref_const / r_max)
        by_pct = n_pct * (ref_const / r_pct)
        print(f'same headroom applied to {input_name}:')
        print(f'  from global max : {by_max:.4f}')
        print(f'  from p{args.percentile}     : {by_pct:.4f}')
        spread = abs(by_max - by_pct) / max(by_max, by_pct)
        if spread > 0.15:
            print(f'  >> the two bases differ by {spread*100:.0f}%: the two '
                  f'distributions differ in shape, not just scale. Pick '
                  f'deliberately rather than taking the average.')
        rec = 0.5 * (by_max + by_pct)

    print(f'\nPaste into include/params.yml under `normalization:`\n')
    print(f'  {input_name}:')
    print(f'    {img_size}:')
    print(f'      max_h: {rec:.3f}')
    print(f'      max_l: {rec:.3f}')

    clipped = n_max / rec
    print(f'\nglobal max / constant = {clipped:.2f} '
          f'({"nothing clips" if clipped <= 1 else "the brightest voxels clip past +1"})')


if __name__ == '__main__':
    main()
