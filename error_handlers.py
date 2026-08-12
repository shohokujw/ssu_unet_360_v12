"""
Error handling utilities for the CT reconstruction project.

Provides consistent error handling, validation, and user-friendly error messages.
"""
import os
import pickle
import torch
import logging
from typing import Any, Dict, Optional


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class FileIOError(Exception):
    """Raised when file I/O operations fail."""
    pass


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass


class GPUError(Exception):
    """Raised when GPU-related operations fail."""
    pass


def safe_load_pickle(file_path: str) -> Dict[str, Any]:
    """
    Safely load a pickle file with error handling.

    Args:
        file_path (str): Path to the pickle file

    Returns:
        dict: Loaded pickle data

    Raises:
        FileIOError: If file cannot be loaded
    """
    if not os.path.exists(file_path):
        raise FileIOError(
            f"Pickle file not found: {file_path}\n"
            f"Please check if the file path is correct."
        )

    try:
        with open(file_path, 'rb') as fid:
            data = pickle.load(fid)
        logger.debug(f"Successfully loaded: {file_path}")
        return data
    except pickle.UnpicklingError as e:
        raise FileIOError(
            f"Corrupted pickle file: {file_path}\n"
            f"The file may be damaged or incompatible.\n"
            f"Error: {str(e)}"
        )
    except Exception as e:
        raise FileIOError(
            f"Failed to load pickle file: {file_path}\n"
            f"Error: {str(e)}"
        )


def safe_save_pickle(data: Dict[str, Any], file_path: str) -> None:
    """
    Safely save data to a pickle file with error handling.

    Args:
        data (dict): Data to save
        file_path (str): Path to save the pickle file

    Raises:
        FileIOError: If file cannot be saved
    """
    # Ensure directory exists
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as e:
            raise FileIOError(
                f"Failed to create directory: {directory}\n"
                f"Error: {str(e)}"
            )

    try:
        with open(file_path, 'wb') as fid:
            pickle.dump(data, fid)
        logger.debug(f"Successfully saved: {file_path}")
    except Exception as e:
        raise FileIOError(
            f"Failed to save pickle file: {file_path}\n"
            f"Error: {str(e)}"
        )


def validate_gpu_availability(gpu_id: int) -> torch.device:
    """
    Validate GPU availability and return appropriate device.

    Args:
        gpu_id (int): GPU ID to use

    Returns:
        torch.device: Device to use for computation

    Raises:
        GPUError: If specified GPU is not available
    """
    if not torch.cuda.is_available():
        logger.warning(
            "CUDA is not available. Falling back to CPU.\n"
            "Training/inference will be significantly slower."
        )
        return torch.device('cpu')

    num_gpus = torch.cuda.device_count()
    if gpu_id >= num_gpus:
        raise GPUError(
            f"GPU {gpu_id} is not available.\n"
            f"System has {num_gpus} GPU(s) (IDs: 0-{num_gpus-1}).\n"
            f"Please specify a valid GPU ID using --gpu_id argument."
        )

    # Check GPU memory
    try:
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
        mem_free = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        logger.info(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(device)}")
        logger.info(f"Total GPU memory: {mem_free:.2f} GB")

        if mem_free < 4.0:
            logger.warning(
                f"GPU {gpu_id} has only {mem_free:.2f} GB memory.\n"
                f"Training with 512×512 images may cause out-of-memory errors.\n"
                f"Consider using smaller image size (--img_size 256 or 128)."
            )

        return device
    except Exception as e:
        raise GPUError(
            f"Failed to initialize GPU {gpu_id}.\n"
            f"Error: {str(e)}"
        )


def validate_config_params(params: Dict[str, Any], required_keys: list) -> None:
    """
    Validate that required configuration parameters exist.

    Args:
        params (dict): Configuration parameters
        required_keys (list): List of required parameter keys

    Raises:
        ConfigurationError: If required parameters are missing
    """
    missing_keys = []
    for key in required_keys:
        if '.' in key:  # Nested key like 'dataset_split.train'
            keys = key.split('.')
            temp = params
            for k in keys:
                if k not in temp:
                    missing_keys.append(key)
                    break
                temp = temp[k]
        else:
            if key not in params:
                missing_keys.append(key)

    if missing_keys:
        raise ConfigurationError(
            f"Missing required configuration parameters:\n"
            f"{', '.join(missing_keys)}\n"
            f"Please check include/params.yml file."
        )


def validate_image_size(img_size: int, supported_sizes: list = [128, 256, 512]) -> None:
    """
    Validate image size parameter.

    Args:
        img_size (int): Image size to validate
        supported_sizes (list): List of supported image sizes

    Raises:
        ValueError: If image size is not supported
    """
    if img_size not in supported_sizes:
        raise ValueError(
            f"Unsupported image size: {img_size}\n"
            f"Supported sizes: {supported_sizes}\n"
            f"Use --img_size argument to specify (e.g., --img_size 512)"
        )


def validate_data_directory(data_dir: str, set_name: str = None) -> None:
    """
    Validate that data directory exists and contains expected data.

    Args:
        data_dir (str): Path to data directory
        set_name (str, optional): Dataset name (train/val/test)

    Raises:
        FileIOError: If directory doesn't exist or is empty
    """
    if not os.path.exists(data_dir):
        raise FileIOError(
            f"Data directory not found: {data_dir}\n"
            f"Please run the preprocessing steps first:\n"
            f"1. step1b_make_fbp_interp.py (or step1a_make_fbp.py for fbp_lh)\n"
            f"2. step1c_calibrate_norm.py\n"
            f"3. step4_Dataset_Fin.py"
        )

    if not os.path.isdir(data_dir):
        raise FileIOError(
            f"Path is not a directory: {data_dir}"
        )

    # Check if directory is empty
    if not os.listdir(data_dir):
        raise FileIOError(
            f"Data directory is empty: {data_dir}\n"
            f"Please run preprocessing steps to generate data."
        )

    logger.debug(f"Data directory validated: {data_dir}")


def check_checkpoint_exists(ckpt_dir: str, epoch: Optional[int] = None) -> bool:
    """
    Check if checkpoint file exists.

    Args:
        ckpt_dir (str): Checkpoint directory
        epoch (int, optional): Specific epoch to check

    Returns:
        bool: True if checkpoint exists

    Raises:
        FileIOError: If checkpoint directory doesn't exist
    """
    if not os.path.exists(ckpt_dir):
        raise FileIOError(
            f"Checkpoint directory not found: {ckpt_dir}\n"
            f"For inference, you need to train the model first using:\n"
            f"python3 step5_Train_FinModel.py"
        )

    if epoch is not None:
        ckpt_file = os.path.join(ckpt_dir, f"model_epoch{epoch:04d}.pth")
        if not os.path.exists(ckpt_file):
            available = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
            raise FileIOError(
                f"Checkpoint not found: {ckpt_file}\n"
                f"Available checkpoints: {available if available else 'None'}"
            )
        return True

    # Check if any checkpoint exists
    checkpoints = [f for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
    if not checkpoints:
        raise FileIOError(
            f"No checkpoints found in: {ckpt_dir}\n"
            f"Please train the model first."
        )

    return True


def safe_create_directory(directory: str) -> None:
    """
    Safely create directory with error handling.

    Args:
        directory (str): Directory path to create

    Raises:
        FileIOError: If directory cannot be created
    """
    try:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Directory created/verified: {directory}")
    except OSError as e:
        raise FileIOError(
            f"Failed to create directory: {directory}\n"
            f"Check permissions and disk space.\n"
            f"Error: {str(e)}"
        )


def log_system_info():
    """Log system and environment information."""
    logger.info("="*50)
    logger.info("System Information")
    logger.info("="*50)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    logger.info("="*50)
