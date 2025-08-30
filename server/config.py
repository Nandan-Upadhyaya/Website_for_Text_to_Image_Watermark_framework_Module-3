"""
Configuration module for AI-Image-Suite server.
Uses relative paths and environment variables for cross-platform compatibility.
"""
import os
from pathlib import Path

# Get the project root directory (parent of server directory)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Default paths relative to project root
DEFAULT_DF_GAN_PATH = PROJECT_ROOT.parent / "DF-GAN"
DEFAULT_AI_SUITE_ROOT = PROJECT_ROOT

def get_path_from_env(env_var: str, default_path: Path, description: str = "") -> Path:
    """
    Get a path from environment variable or use default.
    
    Args:
        env_var: Environment variable name
        default_path: Default path if env var not set
        description: Human readable description for error messages
    
    Returns:
        Path object
    """
    env_value = os.getenv(env_var)
    if env_value:
        return Path(env_value).absolute()
    return default_path.absolute()

# Core paths
DF_GAN_PATH = get_path_from_env('DF_GAN_PATH', DEFAULT_DF_GAN_PATH, "DF-GAN repository")
AI_SUITE_ROOT = get_path_from_env('AI_SUITE_ROOT', DEFAULT_AI_SUITE_ROOT, "AI-Image-Suite root")

# Derived paths
DF_GAN_CODE_PATH = DF_GAN_PATH / 'code'
DF_GAN_SRC_PATH = DF_GAN_CODE_PATH / 'src'

# Model weights paths
CUB_WEIGHTS = get_path_from_env('CUB_WEIGHTS', AI_SUITE_ROOT / 'models' / 'CUB.pth', "CUB model weights")
COCO_WEIGHTS = get_path_from_env('COCO_WEIGHTS', AI_SUITE_ROOT / 'models' / 'COCO.pth', "COCO model weights")

# Data directories
def get_data_dir(dataset: str) -> Path:
    """Get data directory for a specific dataset."""
    if dataset.lower() in ['cub', 'birds', 'bird']:
        return DF_GAN_PATH / 'data' / 'birds'
    elif dataset.lower() == 'coco':
        return DF_GAN_PATH / 'data' / 'coco'
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def validate_paths():
    """
    Validate that all required paths exist.
    
    Returns:
        tuple: (success: bool, issues: list)
    """
    issues = []
    
    # Check DF-GAN path
    if not DF_GAN_PATH.exists():
        issues.append(f"DF-GAN repository not found at {DF_GAN_PATH}")
    elif not DF_GAN_CODE_PATH.exists():
        issues.append(f"DF-GAN code directory not found at {DF_GAN_CODE_PATH}")
    elif not DF_GAN_SRC_PATH.exists():
        issues.append(f"DF-GAN src directory not found at {DF_GAN_SRC_PATH}")
    elif not (DF_GAN_SRC_PATH / 'sample.py').exists():
        issues.append(f"DF-GAN sample.py not found at {DF_GAN_SRC_PATH / 'sample.py'}")
    
    # Check model weights
    if not CUB_WEIGHTS.exists():
        issues.append(f"CUB weights not found at {CUB_WEIGHTS}")
    
    if not COCO_WEIGHTS.exists():
        issues.append(f"COCO weights not found at {COCO_WEIGHTS}")
    
    # Check data directories
    birds_data = get_data_dir('birds')
    if not birds_data.exists():
        issues.append(f"Birds data directory not found at {birds_data}")
    
    coco_data = get_data_dir('coco')
    if not coco_data.exists():
        issues.append(f"COCO data directory not found at {coco_data}")
    
    return len(issues) == 0, issues

def get_config_info():
    """Get current configuration for debugging."""
    return {
        'PROJECT_ROOT': str(PROJECT_ROOT),
        'DF_GAN_PATH': str(DF_GAN_PATH),
        'AI_SUITE_ROOT': str(AI_SUITE_ROOT),
        'DF_GAN_CODE_PATH': str(DF_GAN_CODE_PATH),
        'DF_GAN_SRC_PATH': str(DF_GAN_SRC_PATH),
        'CUB_WEIGHTS': str(CUB_WEIGHTS),
        'COCO_WEIGHTS': str(COCO_WEIGHTS),
        'birds_data': str(get_data_dir('birds')),
        'coco_data': str(get_data_dir('coco')),
    }
