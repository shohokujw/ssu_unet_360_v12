"""
File and path utility functions for the CT reconstruction project.
"""
import os
from typing import Tuple


def extract_subject_id(file_path: str) -> str:
    """
    Extract subject ID from file path.

    Assumes filename format: prefix_XXXX.pkl where XXXX is the subject ID.

    Args:
        file_path (str): Full path to file

    Returns:
        str: Subject ID (e.g., '0004')

    Example:
        >>> extract_subject_id('/path/to/FBP_0004.pkl')
        '0004'
    """
    filename = os.path.basename(file_path)
    # Remove extension and get last part after underscore
    subject_id = filename.replace('.pkl', '').split('_')[-1]
    return subject_id


def get_dataset_directory(subject_id: str,
                          train_subs: list,
                          val_subs: list,
                          test_subs: list,
                          train_dir: str,
                          val_dir: str,
                          test_dir: str) -> str:
    """
    Determine which dataset directory (train/val/test) a subject belongs to.

    Args:
        subject_id (str): Subject ID to classify
        train_subs (list): List of training subject IDs
        val_subs (list): List of validation subject IDs
        test_subs (list): List of test subject IDs
        train_dir (str): Training directory path
        val_dir (str): Validation directory path
        test_dir (str): Test directory path

    Returns:
        str: Directory path for the subject

    Raises:
        ValueError: If subject not found in any split
    """
    if subject_id in train_subs:
        return train_dir
    elif subject_id in val_subs:
        return val_dir
    elif subject_id in test_subs:
        return test_dir
    else:
        raise ValueError(
            f"Subject {subject_id} not found in any dataset split.\n"
            f"Please check the dataset_split configuration in params.yml"
        )


def create_model_directories(base_dir: str,
                            model_name: str,
                            data_name: str,
                            experiment_name: str,
                            img_size: int) -> Tuple[str, str, str]:
    """
    Create standardized directory structure for model training.

    Creates:
        - checkpoint directory: base_dir/model_name/data_name/exp/ckpt/img_size/
        - log directory: base_dir/model_name/data_name/exp/log/img_size/
        - result directory: base_dir/model_name/data_name/exp/result/img_size/

    Args:
        base_dir (str): Base directory (project root)
        model_name (str): Model name (e.g., 'Pix2Pix')
        data_name (str): Dataset name (e.g., 'model_Gen3_7')
        experiment_name (str): Experiment identifier (e.g., 'ndG7_ndD4_dropout_3')
        img_size (int): Image size (128, 256, or 512)

    Returns:
        tuple: (ckpt_dir, log_dir, result_dir)
    """
    exp_base = os.path.join(base_dir, model_name, data_name, experiment_name)

    ckpt_dir = os.path.join(exp_base, 'ckpt', str(img_size))
    log_dir = os.path.join(exp_base, 'log', str(img_size))
    result_dir = os.path.join(exp_base, 'result', str(img_size))

    return ckpt_dir, log_dir, result_dir


def get_experiment_name(model_name: str,
                       num_down_G: int = None,
                       num_down_D: int = None,
                       num_down: int = None,
                       dropout: float = 0.3) -> str:
    """
    Generate standardized experiment name from model configuration.

    Args:
        model_name (str): Model name ('Pix2Pix' or 'UNet')
        num_down_G (int, optional): Generator down-sampling levels (for Pix2Pix)
        num_down_D (int, optional): Discriminator down-sampling levels (for Pix2Pix)
        num_down (int, optional): Down-sampling levels (for UNet)
        dropout (float): Dropout rate

    Returns:
        str: Experiment name (e.g., 'ndG7_ndD4_dropout_3')
    """
    if model_name == 'Pix2Pix':
        if num_down_G is None or num_down_D is None:
            raise ValueError("Pix2Pix requires num_down_G and num_down_D")
        return f"ndG{num_down_G}_ndD{num_down_D}_dropout_{int(10*dropout)}"
    elif model_name == 'UNet':
        if num_down is None:
            raise ValueError("UNet requires num_down")
        return f"nd{num_down}_dropout_{int(10*dropout)}"
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def get_data_directory(base_dir: str,
                       data_name: str,
                       subset: str,
                       img_size: int) -> str:
    """
    Get standardized data directory path.

    Args:
        base_dir (str): Base directory (project root)
        data_name (str): Dataset name
        subset (str): Dataset subset ('train', 'val', or 'test')
        img_size (int): Image size

    Returns:
        str: Data directory path
    """
    if subset not in ['train', 'val', 'test']:
        raise ValueError(f"Invalid subset: {subset}. Must be 'train', 'val', or 'test'")

    return os.path.join(base_dir, 'datasets', data_name, 'fin_set', str(img_size), subset)
