"""
Dataset Preparation Script (Step 3)

This script prepares the final training/validation/test datasets by:
1. Loading FBP reconstructions (9-view sparse) and ground truth (720-view full)
2. Combining dual-energy data (low + high energy weighted)
3. Normalizing the data based on configuration
4. Splitting into train/val/test sets
5. Saving preprocessed data for model training

Input:
    - FBP reconstructions from step2 (sparse-view)
    - Full-view reconstructions (ground truth labels)

Output:
    - Preprocessed pickle files in datasets/<model_name>/fin_set/<img_size>/{train,val,test}/

Usage:
    python step3_Dataset_Fin.py --img_size 512
"""

import os
import sys
from glob import glob
from preprocessing import normalize
from config_utils import load_ct_params, get_dataset_split, get_normalization_params
from error_handlers import (
    safe_load_pickle, safe_save_pickle, safe_create_directory,
    validate_image_size, logger
)
from file_utils import extract_subject_id, get_dataset_directory, get_data_directory

current_folder = os.path.dirname(os.path.realpath(__file__))

# Load configuration with error handling
try:
    params = load_ct_params()
    logger.info("Configuration loaded successfully")
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    sys.exit(1)
model_name = params['model_name']

fbp_path = params['fbp_path']
label_path = f'{current_folder}/datasets/{model_name}/fbp_full'
label_name = 'FBPfull512_720view'
label_key = 'fbp'

w_l = params['w_l']
w_h = params['w_h']

data_root = f"{current_folder}/datasets/"
if os.path.isabs(label_path):
    label_dir = label_path
else:
    label_dir = f'{current_folder}/datasets/{model_name}/{label_path}'

if os.path.isabs(fbp_path):
    fbp_dir = fbp_path
else:
    fbp_dir = os.path.join(data_root,fbp_path)

fbp_dir = os.path.join(data_root,model_name,fbp_path)

import argparse

parser = argparse.ArgumentParser()

size = params['img_mat'][0]
parser.add_argument('--img_size', type=int,default=size)

args = parser.parse_args()

img_size = args.img_size # 128 or 256 or 512

# Validate image size
try:
    validate_image_size(img_size)
except ValueError as e:
    logger.error(str(e))
    sys.exit(1)

# Use utility function for standardized directory paths
train_dir = get_data_directory(current_folder, model_name, 'train', img_size)
val_dir = get_data_directory(current_folder, model_name, 'val', img_size)
test_dir = get_data_directory(current_folder, model_name, 'test', img_size)

# Create directories with error handling
try:
    safe_create_directory(train_dir)
    safe_create_directory(val_dir)
    safe_create_directory(test_dir)
except Exception as e:
    logger.error(f"Failed to create output directories: {e}")
    sys.exit(1)

subjects = []
lst_data = sorted(glob(os.path.join(fbp_dir, '*.pkl')))
for path_data in lst_data:
    # Use utility function to extract subject ID
    subject_id = extract_subject_id(path_data)
    subjects.append(subject_id)

print(subjects)

# Load dataset split from configuration
try:
    train_subs, valid_subs, test_subs = get_dataset_split(params)
    logger.info(f"Dataset split: {len(train_subs)} train, {len(valid_subs)} val, {len(test_subs)} test")
except KeyError as e:
    logger.error(str(e))
    sys.exit(1)

# Alternative: random split (commented out, kept for reference)
'''
train_subs,test_subs = train_test_split(subjects,test_size=0.1,random_state=42, shuffle=True)
valid_subs,test_subs = train_test_split(test_subs,test_size=0.5,random_state=42,shuffle=True)
'''

print(len(subjects))
print(len(train_subs))
print(len(valid_subs))
print(len(test_subs))

lst_data_label = sorted(glob(os.path.join(label_dir, '*.pkl')))
lst_data_fbp   = sorted(glob(os.path.join(fbp_dir, '*.pkl')))

if not lst_data_fbp:
    logger.error(f"No FBP data files found in: {fbp_dir}")
    logger.error("Please run step2_scale_back_and_run_FBP.py first.")
    sys.exit(1)

# Get normalization parameters from configuration
try:
    fbp_norm_params = get_normalization_params(params, 'fbp', img_size)
    max_h = fbp_norm_params['max_h']
    max_l = fbp_norm_params['max_l']
    max_lh = w_l * max_l + w_h * max_h
    logger.info(f"FBP normalization params: max_h={max_h}, max_l={max_l}")
except KeyError as e:
    logger.error(str(e))
    sys.exit(1)

for path_data in lst_data_fbp:
    # Use utility function to extract subject ID
    sub = extract_subject_id(path_data)

    # Use utility function to get appropriate directory
    try:
        target_dir = get_dataset_directory(
            sub, train_subs, valid_subs, test_subs,
            train_dir, val_dir, test_dir
        )
    except ValueError as e:
        logger.warning(f"{e} - Skipping subject {sub}")
        continue

    print(path_data)

    # Load pickle file with error handling
    try:
        data = safe_load_pickle(path_data)
        fbp_lh = data['fbp_lh'].copy()
    except Exception as e:
        logger.error(f"Failed to load {path_data}: {e}")
        continue

    fbp_lh = normalize(fbp_lh, max_value=max_h)

    pathTmp = os.path.join(target_dir, sub)
    safe_create_directory(pathTmp)

    n_dimz = len(fbp_lh)
    for zi in range(n_dimz):
        path_save = os.path.join(pathTmp, f'FBP_{sub}_{zi:03}.pkl')
        data_zi = {'fbp_lh': fbp_lh[zi]}
        try:
            safe_save_pickle(data_zi, path_save)
        except Exception as e:
            logger.error(f"Failed to save {path_save}: {e}")
            continue

# Get label normalization parameters from configuration
try:
    label_norm_params = get_normalization_params(params, 'label', img_size)
    max_h = label_norm_params['max_h']
    max_l = label_norm_params['max_l']
    max_lh = w_l * max_l + w_h * max_h
    logger.info(f"Label normalization params: max_h={max_h}, max_l={max_l}")
except KeyError as e:
    logger.error(str(e))
    sys.exit(1)

if not lst_data_label:
    logger.warning(f"No label data files found in: {label_dir}")
    logger.warning("Skipping label processing.")
else:
    for path_data in lst_data_label:
        # Use utility function to extract subject ID
        sub = extract_subject_id(path_data)

        # Use utility function to get appropriate directory
        try:
            target_dir = get_dataset_directory(
                sub, train_subs, valid_subs, test_subs,
                train_dir, val_dir, test_dir
            )
        except ValueError as e:
            logger.warning(f"{e} - Skipping subject {sub}")
            continue

        print(path_data)

        # Load pickle file with error handling
        try:
            data = safe_load_pickle(path_data)
            label_lh = data[f'{label_key}_lh']
        except Exception as e:
            logger.error(f"Failed to load {path_data}: {e}")
            continue

        label_lh[label_lh<0] = 0
        label_lh = normalize(label_lh, max_value=max_lh)

        pathTmp = os.path.join(target_dir, sub)
        safe_create_directory(pathTmp)

        n_dimz = len(label_lh)
        for zi in range(n_dimz):
            path_save = os.path.join(pathTmp, f'{label_name}_{sub}_{zi:03}.pkl')
            data_zi = {f'{label_key}_lh': label_lh[zi]}
            try:
                safe_save_pickle(data_zi, path_save)
            except Exception as e:
                logger.error(f"Failed to save {path_save}: {e}")
                continue

logger.info("Dataset preparation completed successfully!")
    





