import base64
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

# Fix for compatibility with newer Werkzeug versions
import werkzeug
if not hasattr(werkzeug.urls, 'url_quote'):
    werkzeug.urls.url_quote = werkzeug.urls.quote

# Add torch import
import torch

from flask import Flask, jsonify, request
from flask_cors import CORS

# Import the wrapper module
from dfgan_wrapper import DFGANGenerator

app = Flask(__name__)
CORS(app)

# Paths
AI_SUITE_ROOT = Path(r'c:\Users\nanda\OneDrive\Desktop\AI-Image-Suite')
DF_GAN_PATH = Path(os.getenv('DF_GAN_PATH', r'c:\Users\nanda\OneDrive\Desktop\DF-GAN'))
DF_GAN_CODE_PATH = DF_GAN_PATH / 'code'

# Use weights from AI-Image-Suite\models as requested
CUB_WEIGHTS = Path(os.getenv('CUB_WEIGHTS', str(AI_SUITE_ROOT / 'models' / 'CUB.pth')))
COCO_WEIGHTS = Path(os.getenv('COCO_WEIGHTS', str(AI_SUITE_ROOT / 'models' / 'COCO.pth')))

def is_cub_prompt(prompt: str) -> bool:
  p = (prompt or '').lower()
  keys = ['bird', 'sparrow', 'eagle', 'jay', 'owl', 'finch', 'feather', 'beak', 'wing', 'robin', 'parrot']
  return any(k in p for k in keys)

def ensure_paths_ok():
    if not DF_GAN_PATH.exists():
        raise FileNotFoundError(f'DF-GAN repo path not found at {DF_GAN_PATH}')
    if not DF_GAN_CODE_PATH.exists():
        raise FileNotFoundError(f'DF-GAN code directory not found at {DF_GAN_CODE_PATH}')
    if not CUB_WEIGHTS.exists():
        raise FileNotFoundError(f'CUB weights not found at {CUB_WEIGHTS}')
    if not COCO_WEIGHTS.exists():
        raise FileNotFoundError(f'COCO weights not found at {COCO_WEIGHTS}')
    
    # Check for sample.py script in the correct location
    sample_script = DF_GAN_CODE_PATH / 'src' / 'sample.py'
    if not sample_script.exists():
        raise FileNotFoundError(f'Sample script not found at {sample_script}')
    
    return sample_script

# Modify the dfgan_generate function to use our wrapper
def dfgan_generate(prompt: str, model_key: str, out_dir: Path, seed: Optional[int]) -> Path:
    """
    Execute DF-GAN inference using our wrapper for sample.py
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Set model-specific paths
    if model_key == 'CUB':
        weights = CUB_WEIGHTS
        data_dir = str(DF_GAN_PATH / 'data' / 'birds')  # Changed from 'bird' to 'birds'
    else:
        weights = COCO_WEIGHTS
        data_dir = str(DF_GAN_PATH / 'data' / 'coco')
    
    print(f"Using model: {weights}")
    print(f"Using data directory: {data_dir}")
    print(f"Prompt: {prompt}")
    
    try:
        # Create or get the generator for this model
        import torch
        model_key_lower = model_key.lower()
        use_cuda = torch.cuda.is_available()
        seed_value = seed if seed is not None and seed >= 0 else 100
        
        # Create generator for this specific model
        generator = DFGANGenerator(
            model_path=str(weights),
            data_dir=data_dir,
            use_cuda=use_cuda,
            seed=seed_value
        )
        
        # Generate the image
        img_path = generator.generate_image(prompt, out_dir)
        print(f"Image generated at: {img_path}")
        
        return img_path
        
    except Exception as e:
        error_msg = f"Error generating image: {str(e)}"
        print(error_msg)
        import traceback
        print(traceback.format_exc())
        raise RuntimeError(error_msg)

def encode_b64(path: Path) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

@app.route('/api/check', methods=['GET'])
def check_setup():
    """Endpoint to check if the setup is correct"""
    issues = []
    paths = {}
    
    # Check DF-GAN path
    if not DF_GAN_PATH.exists():
        issues.append(f"DF-GAN repo not found at {DF_GAN_PATH}")
    else:
        paths['df_gan'] = str(DF_GAN_PATH)
        
        # Check for code directory
        if not DF_GAN_CODE_PATH.exists():
            issues.append(f"Code directory not found at {DF_GAN_CODE_PATH}")
        else:
            paths['df_gan_code'] = str(DF_GAN_CODE_PATH)
            
            # Check for sample.py script
            sample_script = DF_GAN_CODE_PATH / 'src' / 'sample.py'
            if not sample_script.exists():
                issues.append(f"Sample script not found at {sample_script}")
            else:
                paths['sample_script'] = str(sample_script)
    
    # Check model weights
    if not CUB_WEIGHTS.exists():
        issues.append(f"CUB weights not found at {CUB_WEIGHTS}")
    else:
        paths['cub_weights'] = str(CUB_WEIGHTS)
    
    if not COCO_WEIGHTS.exists():
        issues.append(f"COCO weights not found at {COCO_WEIGHTS}")
    else:
        paths['coco_weights'] = str(COCO_WEIGHTS)
    
    # Return status
    if issues:
        return jsonify({
            'status': 'error',
            'issues': issues,
            'paths': paths
        }), 400
    
    return jsonify({
        'status': 'ok',
        'paths': paths
    })

@app.route('/api/generate', methods=['POST'])
def generate():
    payload = request.get_json(force=True, silent=True) or {}
    prompt = (payload.get('prompt') or '').strip()
    dataset = (payload.get('dataset') or '').strip()
    batch_size = int(payload.get('batchSize') or 1)
    seed = int(payload.get('seed') or -1)

    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    try:
        script_path = ensure_paths_ok()
    except Exception as e:
        return jsonify({'error': str(e)}), 400

    # If dataset is 'bird', force CUB model, otherwise check prompt
    if dataset == 'bird':
        model_key = 'CUB'
    elif dataset == 'coco':
        model_key = 'COCO'
    else:
        model_key = 'CUB' if is_cub_prompt(prompt) else 'COCO'

    print(f"Using {model_key} model for prompt: '{prompt}'")
    print(f"Using script: {script_path}")

    tmp_root = Path(tempfile.gettempdir()) / f'dfgan_{uuid.uuid4().hex}'
    images_b64: List[str] = []
    try:
        for _ in range(batch_size):
            img = dfgan_generate(prompt, model_key, tmp_root, seed if seed >= 0 else None)
            images_b64.append(encode_b64(img))
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        shutil.rmtree(tmp_root, ignore_errors=True)
        return jsonify({'error': str(e)}), 500

    shutil.rmtree(tmp_root, ignore_errors=True)
    return jsonify({'images': images_b64, 'model': model_key})

if __name__ == '__main__':
    port = int(os.getenv('PORT', '5001'))
    app.run(host='0.0.0.0', port=port, debug=True)

