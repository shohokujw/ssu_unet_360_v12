"""
Model Training Script (Step 5)

v12: the input is an FBP of network-synthesised views, so the v11 weights are a
good starting point but the input distribution has moved. `--init_from` seeds a
fresh run with another checkpoint's parameters -- weights only, not the
optimiser moments (they encode gradients for the old inputs) and not the epoch
counter (the LR schedule should start over). Resuming this run's own ckpt_dir
still takes precedence, so an interrupted fine-tune continues correctly.

The experiment directory carries `input_name`, so a v12 run never overwrites the
v11 baseline and the two can be compared directly.

This script trains the Pix2Pix GAN model for sparse-view CT reconstruction.

Training Process:
1. Load preprocessed training and validation datasets
2. Initialize Pix2Pix model (Generator + Discriminator)
3. Apply data augmentation (random flips, rotation)
4. Train with adversarial + L1 loss
5. Save checkpoints and log metrics to TensorBoard

Model Architecture:
    - Generator: U-Net with skip connections (7 down-sampling levels)
    - Discriminator: PatchGAN (4 down-sampling levels)
    - Loss: Adversarial loss + L1 reconstruction loss

Input:
    - Preprocessed datasets from step4

Output:
    - Model checkpoints: Pix2Pix/<model_name>/<exp>/ckpt/<img_size>/
    - Training logs: Pix2Pix/<model_name>/<exp>/log/<img_size>/
    - Results: Pix2Pix/<model_name>/<exp>/result/<img_size>/

Usage:
    python3 step5_Train_FinModel.py --img_size 512 --num_downs_G 7 --num_downs_D 4 --gpu_id 0

    # Pix2Pix (nohup)
    nohup python3 -u step5_Train_FinModel.py --img_size 512 --num_downs_G 7 --num_downs_D 4 --model_name Pix2Pix --gpu_id 0 > nohup.out 2>&1 &

    # UNet (nohup)
    nohup python3 -u step5_Train_FinModel.py --img_size 512 --num_downs 3 --model_name UNet --gpu_id 0 > nohup.out 2>&1 &

Watching a nohup'd run:
    # Live progress bar, exactly as if it were in the foreground. tqdm redraws
    # with carriage returns, so do NOT pipe this through tr -- translating them
    # to newlines unrolls the bar into one line per iteration (useful for
    # grepping the log, useless for watching it). And -n 0 starts at the end:
    # without it tail replays 10 "lines", each a whole epoch's worth of redraws.
    tail -n 0 -f nohup.out

    # Epoch, ETA, losses and checkpoint state in one shot
    python3 check_train.py           # --watch 30 to refresh

    # Loss curves and the fixed-slice previews
    open http://localhost:7010       # TensorBoard, IMAGES tab
"""

import yaml
import sys
import os

# Add relative paths to sys.path for module imports
current_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(current_folder, 'Pix2Pix'))

from Pix2Pix.train import Model as Pix2PixModel
from Pix2Pix.train import ModelUNet as UNetModel
from config_utils import load_ct_params, get_batch_size
from error_handlers import (
    validate_gpu_availability, validate_image_size,
    validate_data_directory, safe_create_directory, log_system_info, logger
)
from file_utils import create_model_directories, get_experiment_name
import torch

from torch.utils.data import DataLoader
from include.dataloader import PickelDatasetv2, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, Dual_img_Compose
import argparse
import subprocess

from torch.utils.tensorboard import SummaryWriter

def save_params_to_file(params_dict, filename="params.txt"):
    """Save training parameters to a formatted text file for record-keeping."""
    with open(filename, 'wt') as f:
        border = '╔' + '═' * 35 + '╗\n'
        separator = '╠' + '═' * 35 + '╣\n'
        footer = '╚' + '═' * 35 + '╝\n'

        f.write(border)
        f.write('║{:^35}║\n'.format("PARAMETER TABLE"))
        f.write(separator)

        for k, v in sorted(params_dict.items()):
            f.write('║ {:<15} : {:>15} ║\n'.format(k, v))

        f.write(footer)

def main(args):

    # Log system information
    log_system_info()

    file_path = f"{current_folder}/include/params_model.yml"

    try:
        with open(file_path, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Model config file not found: {file_path}")
        sys.exit(1)

    img_size = args.img_size

    # Validate image size
    try:
        validate_image_size(img_size)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # Load constants from CT system parameters
    try:
        params_ct = load_ct_params()
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)

    z_slice = params_ct['training']['z_slices']
    save_iter = params_ct['training']['save_interval']
    model_name = args.model_name   #'Pix2Pix' 'UNet'
    
    if model_name == "Pix2Pix":
        config[model_name]['num_down_G'] = args.num_downs_G
        config[model_name]['num_down_D'] = args.num_downs_D
    
    elif model_name== "UNet":
        config[model_name]['num_down'] = args.num_downs
    
    
    else:
        raise ValueError("wrong model")

    # Get batch size from configuration
    try:
        batch_size = get_batch_size(params_ct, img_size)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    data_name = params_ct['model_name']
    input_name = params_ct.get('input_name', 'fbp_lh')

    fbp_path = params_ct['fbp_path']
    label_path = f'{current_folder}/fbp_full'
    label_name = 'FBPfull512_720view'
    label_key = 'fbp'

    dropout = config[model_name]['dropout']

    # Use utility function to generate experiment name
    try:
        exp = get_experiment_name(
            model_name,
            num_down_G=config[model_name].get('num_down_G'),
            num_down_D=config[model_name].get('num_down_D'),
            num_down=config[model_name].get('num_down'),
            dropout=dropout
        )
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # Keep v12 runs in their own tree so the v11 baseline stays intact and the
    # two are directly comparable.
    if input_name != 'fbp_lh':
        exp = f'{exp}__{input_name}'

    config[model_name]['batch_size'] = batch_size
    print(config[model_name])
    print(args)

    # Use utility function to create model directories
    try:
        ckpt_dir, log_dir, result_dir = create_model_directories(
            current_folder, model_name, data_name, exp, img_size
        )
        safe_create_directory(ckpt_dir)
        safe_create_directory(log_dir)
        safe_create_directory(result_dir)
        logger.info(f"Model directories created: {exp}")
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
        sys.exit(1)

    logger.info(f"Network input: {input_name}")
    if args.init_from:
        logger.info(f"Fine-tuning from: {args.init_from}")

    data_dir = f'{current_folder}/datasets/{data_name}/fin_set/{img_size}'

    # Validate data directory exists
    try:
        validate_data_directory(data_dir)
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)
    
    # Save parameters to file
    exp_base = os.path.join(current_folder, model_name, data_name, exp)
    save_params_to_file(config[model_name], f"{exp_base}/params_{img_size}.txt")
    
    
    # Validate GPU and get device
    try:
        device = validate_gpu_availability(args.gpu_id)
        logger.info(f'Using device: {device}')
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)

    writer = SummaryWriter(log_dir=log_dir)
    import pathlib, socket
    tb_path = str(pathlib.Path(sys.executable).parent / 'tensorboard')
    if pathlib.Path(tb_path).exists():
        port = 7010
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', port)) != 0:
                    break
            port += 1
        # Without this TensorBoard keeps only ~10 images per tag, so most
        # of the per-epoch previews would never be viewable in the UI.
        subprocess.Popen([tb_path, '--logdir', log_dir, '--port', str(port),
                          '--samples_per_plugin', 'images=0'])
        logger.info(f'TensorBoard started at http://localhost:{port}')
    else:
        logger.warning(f'tensorboard not found at {tb_path}, skipping launch')

    # Get data augmentation parameters from configuration
    max_rotate_degree = params_ct['training']['max_rotate_degree']
    transform = Dual_img_Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation(degrees=max_rotate_degree)
    ])
    
    dataset_train = PickelDatasetv2(root_dir=data_dir,set_name="train",data_firstname=f"FBP_",label_firstname=f"{label_name}_",data_dicname="fbp",label_dicname=f"{label_key}",z_slice=z_slice,transform=transform)    
    loader_train = DataLoader(dataset_train, batch_size=config[model_name]['batch_size'], shuffle=True,num_workers=16)
 
    dataset_val = PickelDatasetv2(root_dir=data_dir,set_name="val",data_firstname=f"FBP_",label_firstname=f"{label_name}_",data_dicname="fbp",label_dicname=f"{label_key}",z_slice=z_slice)
    loader_val = DataLoader(dataset_val, batch_size=config[model_name]['batch_size'],shuffle=False,num_workers=16)

    
    # 모델 타입에 따라 Trainer 선택
    if model_name == "Pix2Pix":
        Trainer = Pix2PixModel(configs=config[model_name],
                               save_iter=save_iter,
                               ckpt_dir=ckpt_dir, result_dir=result_dir,
                               device=device, gpu_id=args.gpu_id,
                               writer=writer,
                               loader_train=loader_train,
                               loader_val=loader_val,
                               img_size=img_size,
                               z_slice=z_slice,
                               init_from=args.init_from,
                               preview_every=args.preview_every)
    elif model_name == "UNet":
        Trainer = UNetModel(configs=config[model_name],
                            save_iter=save_iter,
                            ckpt_dir=ckpt_dir, result_dir=result_dir,
                            device=device, gpu_id=args.gpu_id,
                            writer=writer,
                            loader_train=loader_train,
                            loader_val=loader_val,
                            img_size=img_size,
                            z_slice=z_slice,
                            init_from=args.init_from,
                            preview_every=args.preview_every)

    Trainer.train()

    writer.close()               
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #parser.add_argument('--img_size', type=int, default='128')
    parser.add_argument('--img_size', type=int, default='512')
    parser.add_argument('--num_downs_G', type=int, default='7')
    parser.add_argument('--num_downs_D', type=int, default='4')
    parser.add_argument('--num_downs', type=int, default='3')
    parser.add_argument('--model_name', type=str, default='Pix2Pix') # Pix2Pix, UNet
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--preview_every', type=int, default=1,
                        help='Save a preview PNG every N epochs (default: '
                             'every epoch). Each one costs two forward passes '
                             'and ~340 KB.')
    parser.add_argument('--init_from', type=str, default=None,
                        help='Checkpoint whose WEIGHTS seed a fresh run '
                             '(fine-tuning). Optimiser state and epoch counter '
                             'are not carried over. Ignored if this run already '
                             'has checkpoints of its own.')
    args = parser.parse_args()
    print(args)
    main(args)
