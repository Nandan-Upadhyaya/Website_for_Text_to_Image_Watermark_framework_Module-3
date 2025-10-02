import base64
import io
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional
import sys
import time
import numpy as np
import cv2
from PIL import Image as PILImage

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

# Add Module-2 to Python path
MODULE2_PATH = Path(__file__).parent.parent.parent / "Module-2"
if str(MODULE2_PATH) not in sys.path:
    sys.path.insert(0, str(MODULE2_PATH))

print(f"🔍 [SERVER] Added Module-2 path: {MODULE2_PATH}")

# Try to import the evaluator
try:
    from image_prompt_evaluator import ImagePromptEvaluator
    EVALUATOR_AVAILABLE = True
    print("✅ [SERVER] ImagePromptEvaluator imported successfully")
except ImportError as e:
    print(f"❌ [SERVER] Failed to import ImagePromptEvaluator: {e}")
    EVALUATOR_AVAILABLE = False

# Import configuration and wrapper module
from config import (
    DF_GAN_PATH, DF_GAN_CODE_PATH, DF_GAN_SRC_PATH,
    CUB_WEIGHTS, COCO_WEIGHTS, get_data_dir, validate_paths, get_config_info
)
from dfgan_wrapper import DFGANGenerator

# Import the vehicle generator
from vehicle_gan import is_vehicle_prompt, generate_vehicle_image

# Import other modules
from stylegan_wrapper import StyleGANGenerator
from blend_utils import blend_images

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])  # Explicit CORS for React app

# Global variables
LAST_GEN = {"ts": 0, "prompt": "", "images": []}
stylegan_gen = StyleGANGenerator()
_evaluator = None

# Weak entities for blending
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

def get_evaluator():
    global _evaluator
    if EVALUATOR_AVAILABLE and _evaluator is None:
        try:
            _evaluator = ImagePromptEvaluator()
            print("✅ [SERVER] ImagePromptEvaluator initialized")
        except Exception as e:
            print(f"❌ [SERVER] Failed to initialize evaluator: {e}")
            return None
    return _evaluator

def detect_entity(prompt: str):
    for k, v in weak_entities.items():
        if k in prompt.lower():
            return v
    return None

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

def dfgan_generate(prompt: str, model_key: str, out_dir: Path, seed: Optional[int], steps: int, guidance: float) -> Path:
    """Execute DF-GAN inference using our wrapper for sample.py"""
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
                output_filename = uuid.uuid4().hex
                img_path, vehicle_type = generate_vehicle_image(prompt, tmp_root, output_filename)
                
                if img_path and os.path.exists(str(img_path)):
                    images_b64.append(encode_b64(img_path))
                    print(f"Vehicle image generated: {img_path} (type: {vehicle_type})")
                else:
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

                # Load DF-GAN output as numpy array
                dfgan_img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)

                # Check if prompt needs StyleGAN entity
                entity_type = detect_entity(prompt)
                if entity_type:
                    entity_img = stylegan_gen.generate(entity_type, seed=np.random.randint(10000))
                    blended_img = blend_images(dfgan_img, entity_img, position=(64, 64))
                    # Overwrite DF-GAN file with blended image
                    cv2.imwrite(str(img_path), cv2.cvtColor(blended_img, cv2.COLOR_RGB2BGR))

                images_b64.append(encode_b64(img_path))

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        shutil.rmtree(tmp_root, ignore_errors=True)
        return jsonify({'error': str(e)}), 500

    # Publish event, cleanup, response
    try:
        LAST_GEN["ts"] = int(time.time() * 1000)
        LAST_GEN["prompt"] = prompt
        LAST_GEN["images"] = images_b64[:]
    except Exception:
        pass

    shutil.rmtree(tmp_root, ignore_errors=True)
    return jsonify({'images': images_b64, 'model': model_key})

@app.route('/favicon.ico')
def favicon():
    """Return empty response for favicon to avoid 404s"""
    return '', 204

@app.route('/api/gen-events', methods=['GET'])
def gen_events():
    """Returns the last generation event"""
    return jsonify({
        "ts": LAST_GEN.get("ts", 0),
        "prompt": LAST_GEN.get("prompt", ""),
        "images": LAST_GEN.get("images", []),
    })

@app.route('/api/test', methods=['GET', 'POST'])
def test_endpoint():
    """Simple test endpoint to verify API connectivity"""
    print("🔍 [TEST] /api/test endpoint called")
    if request.method == 'POST':
        data = request.get_json()
        print(f"🔍 [TEST] POST data received: {data}")
        return jsonify({'status': 'success', 'method': 'POST', 'data': data})
    else:
        print("🔍 [TEST] GET request received")
        return jsonify({'status': 'success', 'method': 'GET', 'message': 'API is working'})

@app.route('/api/evaluate-image', methods=['POST', 'OPTIONS'])
def evaluate_single_image():
    """Evaluate uploaded image using Module-2 evaluator"""
    print("\n🔍 [EVAL] /api/evaluate-image called")
    print(f"🔍 [EVAL] Request method: {request.method}")
    print(f"🔍 [EVAL] Request headers: {dict(request.headers)}")
    
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        print("🔍 [EVAL] Handling CORS preflight")
        response = jsonify({'message': 'CORS preflight'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        return response
    
    try:
        print("🔍 [EVAL] Processing POST request")
        
        # Check content type
        if not request.is_json:
            print(f"❌ [EVAL] Request is not JSON, content-type: {request.content_type}")
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        if not data:
            print("❌ [EVAL] No JSON data received")
            return jsonify({'error': 'No data provided'}), 400
            
        print(f"🔍 [EVAL] JSON data keys: {list(data.keys())}")
        
        image_b64 = data.get('image', '')
        prompt = data.get('prompt', '').strip()
        threshold = float(data.get('threshold', 0.22))
        
        print(f"🔍 [EVAL] Prompt: '{prompt[:50]}...'")
        print(f"🔍 [EVAL] Threshold: {threshold}")
        print(f"🔍 [EVAL] Image data length: {len(image_b64)}")
        
        if not image_b64 or not prompt:
            print("❌ [EVAL] Missing image or prompt")
            return jsonify({'error': 'Image and prompt required'}), 400

        # Get evaluator
        evaluator = get_evaluator()
        if not evaluator:
            print("❌ [EVAL] Evaluator not available")
            return jsonify({'error': 'Evaluator not available - check server logs'}), 500

        # Decode base64 image
        try:
            if ',' in image_b64:
                image_b64 = image_b64.split(',', 1)[1]
            
            image_bytes = base64.b64decode(image_b64)
            img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
            print(f"✅ [EVAL] Image decoded: {img.size}")
        except Exception as e:
            print(f"❌ [EVAL] Image decode error: {e}")
            return jsonify({'error': f'Invalid image: {e}'}), 400

        # Save to temp file
        tmp_dir = Path(tempfile.gettempdir()) / f'eval_{uuid.uuid4().hex}'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        img_path = tmp_dir / 'image.png'
        img.save(str(img_path))
        print(f"✅ [EVAL] Image saved to: {img_path}")

        # Run Module-2 evaluation
        print("🔍 [EVAL] Running Module-2 evaluation...")
        try:
            result = evaluator.evaluate_image(str(img_path), prompt, threshold)
            print(f"✅ [EVAL] Evaluation complete!")
            print(f"🎯 [EVAL] Result: {result.get('percentage_match', 'N/A')} ({result.get('quality', 'N/A')})")
            
            if 'keyword_analysis' in result:
                print(f"🎯 [EVAL] Keywords: {len(result['keyword_analysis'])} analyzed")
        except Exception as e:
            print(f"❌ [EVAL] Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            result = {'error': str(e)}

        # Cleanup
        try:
            img_path.unlink(missing_ok=True)
            tmp_dir.rmdir()
        except:
            pass

        if 'error' in result:
            print(f"❌ [EVAL] Returning error: {result['error']}")
            return jsonify(result), 500
            
        print("✅ [EVAL] Returning results to frontend")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ [EVAL] Critical error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', '5001'))
    app.run(host='0.0.0.0', port=port, debug=True)



