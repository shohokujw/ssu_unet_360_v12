"""
Model Inference Script (Step 6)

This script performs inference using a trained Pix2Pix model on test data.

Inference Process:
1. Load trained model checkpoint from specified epoch
2. Load test dataset (preprocessed in step4)
3. Generate high-quality reconstructions from sparse-view inputs
4. Save results as pickle files for evaluation

Input:
    - Trained model checkpoint from step5
    - Test dataset from step4

Output:
    - Reconstructed images saved in Pix2Pix/<model_name>/<exp>/result/<img_size>/

Usage:
    # Best checkpoint, test split only
    python3 step6_Infer_FinModel.py --img_size 512 --num_downs_G 7 --num_downs_D 4 \
        --test_epoch best --sets test --gpu_id 0

    # A specific epoch, every split (~46 GB of pickles)
    python3 step6_Infer_FinModel.py --img_size 512 --num_downs_G 7 --num_downs_D 4 --test_epoch 700 --gpu_id 0
"""

import yaml
import sys
import os

# Add relative paths to sys.path for module imports
current_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_folder, 'Pix2Pix'))

from Pix2Pix.train import Model as pixpix
import torch

from torch.utils.data import DataLoader
from include.dataloader import PickelDatasetv2
import argparse
from config_utils import load_ct_params, get_label_max_lh

def main(args):

    file_path = f"{current_folder}/include/params_model.yml"

    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    img_size = args.img_size

    # System 파라미터 불러오기 (model_profiles 자동 적용)
    params = load_ct_params()

    z_slice = params['training']['z_slices']
    save_iter = params['training']['save_interval']
    model_name = args.model_name   #'Pix2Pix' 'UNet'

    data_name = params['model_name'] # ex) 360ct_72view, 360ct_360view

    fbp_path = params['fbp_path']
    label_path = params['label_path']
    label_name = params['label_name']
    label_key = params['label_key']
    
    config[model_name]['num_down_G'] = args.num_downs_G
    config[model_name]['num_down_D'] = args.num_downs_D        
    
    exp = f"ndG{config[model_name]['num_down_G']}_ndD{config[model_name]['num_down_D']}" if model_name == 'Pix2Pix' else f"nd{config[model_name]['num_down']}"

    # v12: a run trained on a non-default input lives in its own tree; keep the
    # suffix in step with step5_Train_FinModel.py or inference silently loads the
    # v11 baseline's weights instead.
    input_name = params.get('input_name', 'fbp_lh')
    exp_suffix = '' if input_name == 'fbp_lh' else f'__{input_name}'
    print(config[model_name])
    print(args)
    
    #ckpt_dir =  f"{current_folder}/{model_name}/{data_name}/{exp}/ckpt/{img_size}"
    ckpt_dir = args.ckpt_dir or \
        f"{current_folder}/{model_name}/{data_name}/{exp}_dropout_3{exp_suffix}/ckpt/{img_size}"

    result_dir = args.result_dir or \
        f"{current_folder}/{model_name}/{data_name}/{exp}_dropout_3{exp_suffix}/result/{img_size}"
    os.makedirs(result_dir, exist_ok=True)

    data_dir = args.data_dir or f'{current_folder}/datasets/{data_name}/fin_set/{img_size}'

    print(f'ckpt   : {ckpt_dir}')
    print(f'data   : {data_dir}')
    print(f'result : {result_dir}')
    #data_dir = f"/mnt/c/work/backup/v6/model_2_512_mbir_random_split/fin_set/{img_size}"

    # Saved reconstructions must invert with the constant step4 normalised the
    # label with, or every physical value is off by a fixed factor.
    label_max_lh = get_label_max_lh(params, img_size)
    print(f'label max_lh = {label_max_lh:.4f}')

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print('device : ',device)

    set_names = args.sets

    for set_name in set_names:

        print(f"runs: {set_name}")

        result_dir_set = os.path.join(result_dir, set_name)
        os.makedirs(result_dir_set, exist_ok=True)

        dataset_test = PickelDatasetv2(root_dir=data_dir,set_name=set_name,data_firstname=f"FBP_",label_firstname=f"{label_name}_",data_dicname="fbp",label_dicname=label_key,z_slice=z_slice)
        loader_test = DataLoader(dataset_test, batch_size=config[model_name]['batch_size'],shuffle=False,num_workers=16)


        Tester = pixpix(configs = config[model_name],
                        save_iter= save_iter,
                        ckpt_dir=ckpt_dir,result_dir=result_dir_set,
                        device = device, gpu_id = args.gpu_id,
                        writer = None, 
                        loader_train = None,
                        loader_val = loader_test,
                        img_size = img_size,
                        z_slice=z_slice,
                        test_epoch=args.test_epoch,
                        label_max_lh=label_max_lh)


        Tester.test()

        
        
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default='512')
    parser.add_argument('--num_downs_G', type=int, default='7')
    parser.add_argument('--num_downs_D', type=int, default='4')
    parser.add_argument('--num_downs', type=int, default='3')
    parser.add_argument('--model_name', type=str, default='Pix2Pix')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--test_epoch', default='best',
                        help='Checkpoint to run: an epoch number, or "best" '
                             '(model_best.pth, the lowest val L1 of the run). '
                             'Default: best.')
    parser.add_argument('--ckpt_dir', type=str, default=None,
                        help='Checkpoint directory to read instead of this '
                             'experiment\'s. Use with --data_dir/--result_dir '
                             'to score another run (e.g. the v11 baseline) '
                             'through this fixed code path rather than its own, '
                             'which denormalises with the wrong constant.')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='fin_set directory to read instead of this '
                             'project\'s. Must be normalised the same way the '
                             'checkpoint was trained on.')
    parser.add_argument('--result_dir', type=str, default=None,
                        help='Where to write reconstructions.')
    parser.add_argument('--sets', nargs='+', default=['train', 'test', 'val'],
                        choices=['train', 'test', 'val'],
                        help='Which splits to reconstruct. Each subject writes '
                             'a ~640 MB pickle, so all three splits is ~46 GB; '
                             'pass "--sets test" to only do what you are about '
                             'to measure.')
    args = parser.parse_args()
    print(args)
    main(args)
