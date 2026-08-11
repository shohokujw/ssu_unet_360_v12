"""Recalibrate the input normalisation constants for the new reconstruction.

Why this exists
---------------
`preprocessing.normalize` maps a volume with

    out = (x / max_value) / std - mean          # std 0.5, mean 1.0
        = 2*x/max_value - 1                     -> [-1, 1] when x in [0, max_value]

so `max_value` sets the input range the network sees. The constants in
`params.yml` under `normalization.fbp` were fitted to 9-VIEW FBP volumes.

A reconstruction from synthesised views is not the same distribution: the long
streaks that produced the bright/dark tail are largely gone, so its upper
percentile sits elsewhere. Reusing the 9-view constant silently squeezes or
clips the new input, and the denoiser then trains on a badly scaled signal --
a failure that shows up as mediocre results rather than an error.

This measures the new distribution on the TRAIN subjects only (using val/test
to pick a normalisation constant would leak) and prints a params.yml block.

Usage
-----
    python step1c_calibrate_norm.py                       # the v12 input
    python step1c_calibrate_norm.py --input_name fbp_lh   # check the v11 one
"""

import argparse
import os
import pickle
from glob import glob

import numpy as np

from config_utils import load_ct_params, get_dataset_split
from file_utils import extract_subject_id

current_folder = os.path.dirname(os.path.realpath(__file__))


def main():
    ap = argparse.ArgumentParser(description='Fit input normalisation constants')
    ap.add_argument('--input_name', type=str, default=None,
                    help='Directory under datasets/<model>/ to measure '
                         '(default: params.yml input_name)')
    ap.add_argument('--percentile', type=float, default=99.9,
                    help='Upper percentile taken as max_value. 100 would let a '
                         'single hot voxel set the scale for the whole dataset.')
    ap.add_argument('--max_subjects', type=int, default=0,
                    help='0 = all train subjects')
    args = ap.parse_args()

    cfg = load_ct_params()
    model_name = cfg['model_name']
    input_name = args.input_name or cfg.get('input_name', 'fbp_interp')
    img_size = cfg['img_mat'][0]

    src = f'{current_folder}/datasets/{model_name}/{input_name}'
    files = sorted(glob(os.path.join(src, '*.pkl')))
    if not files:
        raise FileNotFoundError(f'No volumes in {src}. Run step1b first.')

    train_subs, _, _ = get_dataset_split(cfg)
    train_subs = set(train_subs)
    files = [f for f in files if extract_subject_id(f) in train_subs]
    if args.max_subjects:
        files = files[:args.max_subjects]
    if not files:
        raise RuntimeError('None of the volumes belong to the train split.')

    print(f'{input_name}  ({len(files)} train subjects, {img_size}²)')
    print(f'percentile   {args.percentile}\n')

    per_subject, pooled = [], []
    for f in files:
        with open(f, 'rb') as fid:
            vol = pickle.load(fid)['fbp_lh']
        vol = np.asarray(vol, dtype=np.float32)
        v = vol[vol > 0]                      # background is exactly 0 after clipping
        p = float(np.percentile(v, args.percentile)) if v.size else 0.0
        per_subject.append(p)
        # Subsample so the pooled percentile is affordable across 51 volumes.
        pooled.append(v[::997])
        print(f'  {extract_subject_id(f)}  max {vol.max():.4f}   '
              f'p{args.percentile} {p:.4f}   mean(>0) {v.mean():.4f}')
        del vol, v

    pooled = np.concatenate(pooled)
    p_pool = float(np.percentile(pooled, args.percentile))
    per_subject = np.array(per_subject)

    print(f'\npooled p{args.percentile}      {p_pool:.4f}')
    print(f'per-subject p{args.percentile}  mean {per_subject.mean():.4f}  '
          f'min {per_subject.min():.4f}  max {per_subject.max():.4f}')

    # normalize() divides by max_value, and step4 passes max_h for the input.
    # A single value is what the pipeline consumes, so report the pooled one.
    print('\nPaste into include/params.yml under `normalization:`\n')
    print(f'  {input_name}:')
    print(f'    {img_size}:')
    print(f'      max_h: {p_pool:.3f}')
    print(f'      max_l: {p_pool:.3f}')
    print(f'\n(Existing 9-view constants for comparison: '
          f"{cfg['normalization']['fbp'][img_size]})")

    frac = float((pooled > p_pool).mean())
    print(f'\n{frac*100:.2f}% of non-zero voxels exceed it and will clip past '
          f'+1 after normalisation.')


if __name__ == '__main__':
    main()
