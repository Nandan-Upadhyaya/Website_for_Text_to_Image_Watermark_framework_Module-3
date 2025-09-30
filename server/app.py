import base64
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional
from stylegan_wrapper import StyleGANGenerator
from blend_utils import blend_images
import numpy as np
import cv2
import os

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Fix for compatibility with newer Werkzeug versions
import werkzeug
if not hasattr(werkzeug.urls, 'url_quote'):
    werkzeug.urls.url_quote = werkzeug.urls.quote

# Add torch import
import torch

from flask import Flask, jsonify, request
from flask_cors import CORS

# Import configuration and wrapper module
from config import (
    DF_GAN_PATH, DF_GAN_CODE_PATH, DF_GAN_SRC_PATH,
    CUB_WEIGHTS, COCO_WEIGHTS, get_data_dir, validate_paths, get_config_info
)
from dfgan_wrapper import DFGANGenerator

# Import the vehicle generator
from vehicle_gan import is_vehicle_prompt, generate_vehicle_image

app = Flask(__name__)
CORS(app)

stylegan_gen = StyleGANGenerator()

weak_entities = {
    "man": "human",
    "woman": "human",
    "boy": "human",
    "girl": "human",
    "child": "human",
    "children": "human",
    "dog": "dog",
    "cat": "cat",
    "tiger": "wild",
    "bear": "wild",
    "zebra": "wild",
    "giraffe": "wild",
}

def detect_entity(prompt: str):
    for k, v in weak_entities.items():
        if k in prompt.lower():
            return v
    return None

def save_and_return(img, outdir="outputs", filename="result.png"):
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, filename)
    cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return {"image_path": out_path}


def is_cub_prompt(prompt: str) -> bool:
  p = (prompt or '').lower()
  keys = ['bird', 'sparrow', 'eagle', 'jay', 'owl', 'finch', 'feather', 'beak', 'wing', 'robin', 'parrot']
  return any(k in p for k in keys)

def ensure_paths_ok():
    """Validate that all required paths exist using the config module."""
    success, issues = validate_paths()
    if not success:
        raise FileNotFoundError(f"Path validation failed: {'; '.join(issues)}")
    
    return DF_GAN_SRC_PATH / 'sample.py'

# Modify the dfgan_generate function to use our wrapper
def dfgan_generate(prompt: str, model_key: str, out_dir: Path, seed: Optional[int], steps: int, guidance: float) -> Path:
    """
    Execute DF-GAN inference using our wrapper for sample.py
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Set model-specific paths using config module
    if model_key == 'CUB':
        weights = CUB_WEIGHTS
        data_dir = str(get_data_dir('birds'))
    else:
        weights = COCO_WEIGHTS
        data_dir = str(get_data_dir('coco'))
    
    print(f"Using model: {weights}")
    print(f"Using data directory: {data_dir}")
    print(f"Prompt: {prompt}")
    
    try:
        # Create or get the generator for this model
        model_key_lower = model_key.lower()
        use_cuda = torch.cuda.is_available()
        seed_value = seed if seed is not None and seed >= 0 else 100

        # Create generator for this specific model
        generator = DFGANGenerator(
            model_path=str(weights),
            data_dir=data_dir,
            use_cuda=use_cuda,
            seed=seed_value,
            steps=steps,
            guidance=guidance
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
    success, issues = validate_paths()
    paths = get_config_info()
    
    # Return status
    if not success:
        return jsonify({
            'status': 'error',
            'issues': issues,
            'paths': paths
        }), 400
    
    return jsonify({
        'status': 'ok',
        'paths': paths
    })

@app.route('/api/generate', methods=['POST', 'OPTIONS'])
def generate():
    if request.method == 'OPTIONS':
        # Handle CORS preflight request
        response = jsonify({
            'message': 'CORS preflight response'
        })
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response

    data = request.get_json()
    prompt = data.get('prompt', 'A dog')
    dataset = (data.get('dataset') or '').strip()
    batch_size = int(data.get('batchSize') or 1)
    seeds = data.get('seeds')
    steps = int(data.get('steps') or 50)
    guidance = float(data.get('guidance') or 7.5)

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
        for i in range(batch_size):
          seed = None
          if seeds and isinstance(seeds, list) and i < len(seeds):
           seed = seeds[i]

          # Vehicle specialized handling
          if is_vehicle_prompt(prompt):
              print(f"[Vehicle] Using specialized vehicle generator for: {prompt}")
              # Fix: Unpack the tuple return value correctly
              output_filename = uuid.uuid4().hex
              img_path, vehicle_type = generate_vehicle_image(prompt, tmp_root, output_filename)
              
              if img_path and os.path.exists(str(img_path)):
                  # The image is already saved to disk - just encode it
                  images_b64.append(encode_b64(img_path))
                  print(f"Vehicle image generated: {img_path} (type: {vehicle_type})")
              else:
                  # Fallback to normal DF-GAN if vehicle generation failed
                  print("Vehicle generation failed, falling back to DF-GAN")
                  img_path = dfgan_generate(
                    prompt, model_key, tmp_root,
                    seed if seed is not None and int(seed) >= 0 else None,
                    steps, guidance
                  )
                  images_b64.append(encode_b64(img_path))
          else:
              img_path = dfgan_generate(
                prompt, model_key, tmp_root,
                seed if seed is not None and int(seed) >= 0 else None,
                steps, guidance
              )

              # --- NEW PART STARTS HERE ---
              # Load DF-GAN output as numpy array
              dfgan_img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

              # Check if prompt needs StyleGAN entity
              entity_type = detect_entity(prompt)
              if entity_type:
               entity_img = stylegan_gen.generate(entity_type, seed=np.random.randint(10000))
               blended_img = blend_images(dfgan_img, entity_img, position=(64, 64))

            # Overwrite DF-GAN file with blended image
               cv2.imwrite(str(img_path), cv2.cvtColor(blended_img, cv2.COLOR_RGB2BGR))
              # --- NEW PART ENDS HERE ---

              images_b64.append(encode_b64(img_path))

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

