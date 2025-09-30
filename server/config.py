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

# Router defaults and evaluators
ROUTER_DEFAULTS = {
    "best_of": int(os.getenv("DFGAN_BEST_OF", 6)),          # number of DF-GAN samples per prompt
    "clip_threshold": float(os.getenv("DFGAN_CLIP_THRESHOLD", 0.27)),  # rough CLIP cosine threshold
    "use_clip": os.getenv("DFGAN_USE_CLIP", "true").lower() in ("1", "true", "yes"),
    "min_subject_size_hint": 0.10,                           # heuristic for expected subject fraction (for later detectors)
}

# Subject terms (animals + humans) used for background prompt rewriting
SUBJECT_TERMS = [
    # humans
    "man","woman","boy","girl","person","people","human",
    # common animals
    "dog","cat","horse","cow","sheep","goat","bird","eagle","parrot","pigeon","duck","chicken",
    "lion","tiger","bear","elephant","zebra","giraffe","monkey","ape","panda","kangaroo","deer","fox","wolf",
    "rabbit","hamster","squirrel","mouse","rat","pig","boar",
    # pets/variants
    "puppy","kitten","hound","husky","bulldog","poodle","retriever","lab","labrador","siamese","persian",
]

NEGATIVE_SUBJECT_HINT = "no people, no humans, no animals"

# NEW: Subject provider configuration (pluggable)
SUBJECT_PROVIDER = {
    # modes: 'assets' (RGBA cutouts), 'http' (endpoint), 'cmd' (external script), 'biggan' (generate), 'none'
    "mode": os.getenv("SUBJECT_PROVIDER_MODE", "assets").lower(),
    "assets_dir": os.getenv("SUBJECT_ASSETS_DIR", str(AI_SUITE_ROOT / "subject_assets")),
    "http_endpoint": os.getenv("SUBJECT_PROVIDER_HTTP", ""),
    "cmd_template": os.getenv("SUBJECT_PROVIDER_CMD", ""),
    "timeout_sec": int(os.getenv("SUBJECT_PROVIDER_TIMEOUT", 120)),
    # BigGAN options
    "biggan_model": os.getenv("BIGGAN_MODEL", "biggan-deep-256"),
    "biggan_truncation": float(os.getenv("BIGGAN_TRUNCATION", 0.5)),
    "biggan_samples": int(os.getenv("BIGGAN_SAMPLES", 8)),  # generate N and CLIP re-rank
}

# NEW: Simple compositing defaults
COMPOSITION_DEFAULTS = {
    "subject_height_frac": float(os.getenv("SUBJECT_HEIGHT_FRAC", 0.35)),
    "bottom_margin_frac": float(os.getenv("SUBJECT_BOTTOM_MARGIN_FRAC", 0.02)),
    "place": os.getenv("SUBJECT_PLACE", "center-bottom"),
    "shadow": os.getenv("SUBJECT_SHADOW", "true").lower() in ("1", "true", "yes"),
    "shadow_blur": int(os.getenv("SUBJECT_SHADOW_BLUR", 15)),
    "shadow_offset_y": int(os.getenv("SUBJECT_SHADOW_OFFSET_Y", 10)),
    "shadow_opacity": float(os.getenv("SUBJECT_SHADOW_OPACITY", 0.35)),
    "color_match": os.getenv("SUBJECT_COLOR_MATCH", "true").lower() in ("1", "true", "yes"),
    "use_poisson": os.getenv("USE_POISSON_BLEND", "true").lower() in ("1", "true", "yes"),
    "inpaint_under_subject": os.getenv("INPAINT_UNDER_SUBJECT", "true").lower() in ("1", "true", "yes"),
}

# NEW: Simple BigGAN-only bypass for subject-only prompts
SIMPLE_BIGGAN_ONLY = os.getenv("SIMPLE_BIGGAN_ONLY", "true").lower() in ("1", "true", "yes")
SIMPLE_BIGGAN_ONLY_SUBJECTS = [s.strip().lower() for s in os.getenv("SIMPLE_BIGGAN_ONLY_SUBJECTS", "dog,cat").split(",") if s.strip()]

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
        # router info
        'router_best_of': ROUTER_DEFAULTS["best_of"],
        'router_clip_threshold': ROUTER_DEFAULTS["clip_threshold"],
        'router_use_clip': ROUTER_DEFAULTS["use_clip"],
        # subject/composition info
        'subject_provider_mode': SUBJECT_PROVIDER["mode"],
        'subject_assets_dir': SUBJECT_PROVIDER["assets_dir"],
        'composition_place': COMPOSITION_DEFAULTS["place"],
        # NEW: biggan-only settings
        'simple_biggan_only': SIMPLE_BIGGAN_ONLY,
        'simple_biggan_only_subjects': SIMPLE_BIGGAN_ONLY_SUBJECTS,
    }
