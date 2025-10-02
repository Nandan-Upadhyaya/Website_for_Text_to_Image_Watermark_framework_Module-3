"""
API wrapper for the image prompt evaluator.
Integrates with the web application to provide image-prompt evaluation.
"""
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

# More robust path resolution
SERVER_DIR = Path(__file__).resolve().parent           # ...\AI-Image-Suite\server
AI_IMAGE_SUITE_DIR = SERVER_DIR.parent                 # ...\AI-Image-Suite
REPO_ROOT = AI_IMAGE_SUITE_DIR.parent                   # ...\WatermarkGAN

# Try multiple possible locations for Module-2
MODULE2_CANDIDATES = [
    str(REPO_ROOT / "Module-2"),                        # d:\WatermarkGAN\Module-2
    str(AI_IMAGE_SUITE_DIR / "Module-2"),              # d:\WatermarkGAN\AI-Image-Suite\Module-2
    "D:\\WatermarkGAN\\Module-2",                      # Absolute path fallback
]

print(f"Looking for Module-2 in: {MODULE2_CANDIDATES}")

for p in MODULE2_CANDIDATES:
    if os.path.exists(p):
        print(f"Found Module-2 at: {p}")
        if p not in sys.path:
            sys.path.insert(0, p)
        break
else:
    print("Module-2 directory not found in any candidate location")

# Initialize flag and evaluator placeholder
EVALUATOR_AVAILABLE = False
_evaluator = None

try:
    # Test dependencies first
    import torch
    import clip
    import nltk
    from PIL import Image
    
    # Now import the evaluator
    from image_prompt_evaluator import ImagePromptEvaluator
    EVALUATOR_AVAILABLE = True
    print("✅ All dependencies available, ImagePromptEvaluator loaded successfully")
except ImportError as e:
    missing = str(e).split("'")[1] if "'" in str(e) else str(e)
    print(f"❌ Missing dependency: {missing}")
    
    if "clip" in missing.lower():
        print("💡 Install CLIP with:")
        print("   pip install git+https://github.com/openai/CLIP.git")
    elif "nltk" in missing.lower():
        print("💡 Install NLTK with:")
        print("   pip install nltk")
    
    EVALUATOR_AVAILABLE = False

def evaluate_image(image_path: str, prompt: str) -> Dict[str, Any]:
    """Evaluate an image against its prompt"""
    print("\n" + "-"*60)
    print(f"🔍 [EVAL-API] evaluate_image called")
    print(f"🔍 [EVAL-API] Prompt: '{prompt[:50]}...'")
    print(f"🔍 [EVAL-API] Image path: {image_path}")
    print(f"🔍 [EVAL-API] Image exists: {os.path.exists(image_path)}")
    print(f"🔍 [EVAL-API] EVALUATOR_AVAILABLE: {EVALUATOR_AVAILABLE}")
    
    ev = get_evaluator()
    print(f"🔍 [EVAL-API] Evaluator instance: {ev is not None}")
    
    if not ev:
        print("❌ [EVAL-API] No evaluator available")
        error_result = {
            "error": "Evaluation not available - missing dependencies", 
            "percentage_match": "0.00%",
            "quality": "Error",
            "feedback": "Module-2 evaluator not loaded",
            "keyword_analysis": [],
            "missing_feature_analysis": {
                "missing_features": [],
                "weak_features": [],
                "present_features": []
            },
            "detailed_metrics": {"raw_score": 0.0, "average_score": 0.0}
        }
        print(f"❌ [EVAL-API] Returning error result: {error_result}")
        return error_result
        
    if not os.path.exists(image_path):
        print(f"❌ [EVAL-API] Image file not found: {image_path}")
        return {"error": "Image file not found"}
    
    try:
        print("🔍 [EVAL-API] Calling Module-2 ImagePromptEvaluator.evaluate_image...")
        print(f"🔍 [EVAL-API] Evaluator type: {type(ev)}")
        print(f"🔍 [EVAL-API] Evaluator has evaluate_image method: {hasattr(ev, 'evaluate_image')}")
        
        # Call the actual Module-2 evaluation
        results = ev.evaluate_image(image_path, prompt)
        
        print(f"🔍 [EVAL-API] Module-2 returned type: {type(results)}")
        print(f"🔍 [EVAL-API] Module-2 returned keys: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
        
        if isinstance(results, dict):
            # Verify essential fields are present
            essential_fields = ['percentage_match', 'quality', 'feedback']
            for field in essential_fields:
                if field in results:
                    print(f"✅ [EVAL-API] Found {field}: {results[field]}")
                else:
                    print(f"❌ [EVAL-API] Missing essential field: {field}")
            
            # Check keyword analysis
            if 'keyword_analysis' in results:
                kw_count = len(results['keyword_analysis'])
                print(f"✅ [EVAL-API] Found keyword_analysis with {kw_count} items")
                if kw_count > 0:
                    first_kw = results['keyword_analysis'][0]
                    print(f"✅ [EVAL-API] First keyword sample: {first_kw}")
            else:
                print(f"❌ [EVAL-API] Missing keyword_analysis")
                
            # Ensure we have the exact same structure as Module-2 returns
            print(f"✅ [EVAL-API] Evaluation successful - returning Module-2 results as-is")
            print("-"*60 + "\n")
            return results
        else:
            print(f"❌ [EVAL-API] Module-2 returned non-dict: {results}")
            return {"error": f"Invalid result type from Module-2: {type(results)}"}
        
    except Exception as e:
        import traceback
        print(f"❌ [EVAL-API] Exception during Module-2 evaluation: {e}")
        print("❌ [EVAL-API] Full traceback:")
        traceback.print_exc()
        print("-"*60 + "\n")
        return {"error": str(e)}

def get_evaluator():
    """Get or create evaluator instance"""
    global _evaluator, EVALUATOR_AVAILABLE
    print(f"🔍 [EVAL-API] get_evaluator called. Available: {EVALUATOR_AVAILABLE}")
    
    if not EVALUATOR_AVAILABLE:
        print("❌ [EVAL-API] EVALUATOR_AVAILABLE is False")
        return None
        
    if _evaluator is None:
        print("🔍 [EVAL-API] Creating new evaluator instance...")
        try:
            _evaluator = ImagePromptEvaluator()
            print("✅ [EVAL-API] ImagePromptEvaluator created successfully")
            print(f"✅ [EVAL-API] Evaluator device: {getattr(_evaluator, 'device', 'unknown')}")
        except Exception as e:
            import traceback
            print(f"❌ [EVAL-API] Error creating evaluator: {e}")
            traceback.print_exc()
            _evaluator = None
    else:
        print("✅ [EVAL-API] Using existing evaluator instance")
        
    return _evaluator

def evaluate_image_batch(image_paths: List[str], prompts: List[str]) -> List[Dict[str, Any]]:
    """Evaluate multiple images against their prompts"""
    if len(image_paths) != len(prompts):
        return [{"error": "Number of images and prompts must match",
                 "percentage_match": "0.00%", 
                 "quality": "Error"}]
    
    results = []
    for img, prompt in zip(image_paths, prompts):
        results.append(evaluate_image(img, prompt))
    
    return results
