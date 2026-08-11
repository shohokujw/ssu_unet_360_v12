"""
Configuration utility functions for loading and accessing project settings.
"""
import os
import logging
from op_ct_sstlabs import get_params

logger = logging.getLogger(__name__)


def get_project_root():
    """
    Get the project root directory.

    Returns:
        str: Absolute path to project root
    """
    return os.path.dirname(os.path.realpath(__file__))


def load_ct_params(file_path=None):
    """
    Load CT system parameters from YAML configuration.

    Args:
        file_path (str, optional): Path to params.yml. If None, uses default location.

    Returns:
        dict: CT system parameters

    Raises:
        FileNotFoundError: If configuration file doesn't exist
    """
    if file_path is None:
        root = get_project_root()
        file_path = os.path.join(root, 'include', 'params.yml')

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Configuration file not found: {file_path}\n"
            f"Please ensure include/params.yml exists."
        )

    try:
        params = get_params(file_path=file_path)
        logger.debug(f"Loaded configuration from: {file_path}")
        if 'model_profiles' in params:
            params = resolve_model_params(params)
        return params
    except Exception as e:
        raise RuntimeError(
            f"Failed to load configuration from: {file_path}\n"
            f"Error: {str(e)}"
        )


def resolve_model_params(params):
    """
    Resolve model-specific parameters based on model_name.

    Reads model_profiles[model_name] and merges fbp_path, ang, w_l, w_h
    into top-level params so existing code works without changes.

    Args:
        params (dict): CT system parameters with model_profiles section

    Returns:
        dict: params with model-specific values merged in
    """
    model_name = params['model_name']
    try:
        profile = params['model_profiles'][model_name]
    except KeyError:
        available = list(params.get('model_profiles', {}).keys())
        raise KeyError(
            f"Model profile '{model_name}' not found in model_profiles.\n"
            f"Available profiles: {available}"
        )

    for key, value in profile.items():
        params[key] = value
    logger.debug(f"Resolved model profile: {model_name}")
    return params


def get_dataset_split(params):
    """
    Get train/val/test dataset split from parameters.

    Args:
        params (dict): CT system parameters

    Returns:
        tuple: (train_subjects, val_subjects, test_subjects)

    Raises:
        KeyError: If dataset_split configuration is missing
    """
    try:
        return (
            params['dataset_split']['train'],
            params['dataset_split']['val'],
            params['dataset_split']['test']
        )
    except KeyError as e:
        raise KeyError(
            f"Dataset split configuration missing in params.yml.\n"
            f"Please ensure 'dataset_split' section exists with 'train', 'val', and 'test' keys.\n"
            f"Error: {str(e)}"
        )


def get_normalization_params(params, data_type, img_size):
    """
    Get normalization parameters for specified data type and image size.

    Args:
        params (dict): CT system parameters
        data_type (str): 'fbp' or 'label'
        img_size (int): Image size (128, 256, or 512)

    Returns:
        dict: Dictionary with 'max_h' and 'max_l' keys

    Raises:
        KeyError: If data_type or img_size not found in configuration
    """
    try:
        return params['normalization'][data_type][img_size]
    except KeyError as e:
        raise KeyError(
            f"Normalization parameters not found for data_type='{data_type}', "
            f"img_size={img_size}. Check include/params.yml configuration."
        ) from e


def get_training_params(params):
    """
    Get training-related parameters.

    Args:
        params (dict): CT system parameters

    Returns:
        dict: Training parameters (z_slices, save_interval, batch_sizes, etc.)
    """
    return params['training']


def get_batch_size(params, img_size):
    """
    Get batch size for specified image size.

    Args:
        params (dict): CT system parameters
        img_size (int): Image size (128, 256, or 512)

    Returns:
        int: Batch size

    Raises:
        ValueError: If img_size not supported
    """
    try:
        return params['training']['batch_sizes'][img_size]
    except KeyError:
        supported = list(params['training']['batch_sizes'].keys())
        raise ValueError(
            f"Unsupported image size: {img_size}. "
            f"Supported sizes: {supported}"
        )


# Constants (derived from configuration)
PROJECT_ROOT = get_project_root()
DEFAULT_PARAMS_PATH = os.path.join(PROJECT_ROOT, 'include', 'params.yml')
DEFAULT_MODEL_PARAMS_PATH = os.path.join(PROJECT_ROOT, 'include', 'params_model.yml')
