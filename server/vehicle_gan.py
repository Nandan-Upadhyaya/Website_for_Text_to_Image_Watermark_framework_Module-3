"""
Specialized vehicle generation module using BigGAN.
Provides accurate vehicle images by carefully mapping vehicle types to ImageNet classes.
"""
import time
import torch
import numpy as np
import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from PIL import Image

# Import reusable components
from stylegan_wrapper import (_init_clip, _clip_score, _init_biggan, 
                             _to_rgba_cutout, _biggan_generate)

# Specialized vehicle class mappings based on ImageNet
VEHICLE_CLASS_MAP = {
    # Cars by type
    "car": ["sports car", "convertible", "racer", "cab", "limousine"],
    "sports car": ["sports car", "racer", "convertible"],
    "convertible": ["convertible"],
    "sedan": ["cab", "limousine"],
    "suv": ["minivan", "beach wagon", "station wagon"],
    "race car": ["racer", "sports car"],
    "luxury car": ["limousine", "cab"],
    
    # Trucks by type
    "truck": ["pickup", "trailer truck", "moving van", "tractor", "fire engine"],
    "pickup truck": ["pickup"],
    "pickup": ["pickup"],
    "semi truck": ["trailer truck", "tractor"],
    "fire truck": ["fire engine"],
    "tow truck": ["tow truck"],
    
    # Buses
    "bus": ["school bus", "minibus", "trolleybus"],
    "school bus": ["school bus"],
    
    # Two-wheelers
    "motorcycle": ["motorcycle", "motor scooter"],
    "bike": ["mountain bike", "bicycle-built-for-two"],
    "bicycle": ["mountain bike", "bicycle-built-for-two"],
    "scooter": ["motor scooter", "moped"],
    "moped": ["moped", "motor scooter"],
    
    # Specialized vehicles
    "train": ["passenger car", "steam locomotive", "electric locomotive"],
    "boat": ["speedboat", "lifeboat", "canoe", "yawl"],
    "airplane": ["airliner", "warplane"],
    
    # Generic prompt fallback
    "vehicle": ["sports car", "convertible", "pickup", "motorcycle", "mountain bike"]
}

# Optimal truncation values for different vehicle types
TRUNCATION_MAP = {
    "car": 0.3,
    "sports car": 0.32,
    "convertible": 0.28,
    "truck": 0.35,
    "pickup truck": 0.35,
    "bus": 0.4,
    "motorcycle": 0.3,
    "bike": 0.35,
    "bicycle": 0.35,
    "scooter": 0.4,
    "vehicle": 0.32  # generic fallback
}

# Special CLIP prompts optimized for each vehicle type
CLIP_PROMPTS = {
    "car": "a photo of a realistic car, full vehicle, clear view",
    "sports car": "a photo of a sports car, full vehicle, side view",
    "convertible": "a photo of a blue convertible car, full vehicle",
    "truck": "a photo of a truck, full vehicle, realistic",
    "pickup truck": "a photo of a pickup truck, full vehicle, side view",
    "bus": "a photo of a bus, full vehicle, side view",
    "motorcycle": "a photo of a motorcycle, full vehicle, side view",
    "bike": "a photo of a bicycle, mountain bike, full vehicle",
    "bicycle": "a photo of a bicycle, mountain bike, full vehicle",
    "scooter": "a photo of a motor scooter, full vehicle",
    "vehicle": "a photo of a vehicle, full view, realistic"
}

class VehicleGenerator:
    """Specialized generator for high-quality vehicle images using BigGAN."""
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize the vehicle generator."""
        self.device = device or (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        # Initialize with explicit model name from supported models
        self.model_initialized = _init_biggan(self.device, "biggan-deep-256")
        _init_clip(self.device)
        
    def _get_vehicle_type(self, prompt: str) -> str:
        """Extract the main vehicle type from a prompt."""
        prompt_lower = prompt.lower()
        for vehicle_type in VEHICLE_CLASS_MAP.keys():
            if vehicle_type in prompt_lower:
                return vehicle_type
        return "vehicle"  # Default fallback
    
    def _get_optimal_params(self, vehicle_type: str) -> Tuple[List[str], float, str]:
        """Get optimal parameters for a specific vehicle type."""
        # Get class names
        classes = VEHICLE_CLASS_MAP.get(vehicle_type, VEHICLE_CLASS_MAP["vehicle"])
        
        # Get optimal truncation
        truncation = TRUNCATION_MAP.get(vehicle_type, 0.35)
        
        # Get specialized CLIP prompt
        clip_prompt = CLIP_PROMPTS.get(vehicle_type, CLIP_PROMPTS["vehicle"])
        
        return classes, truncation, clip_prompt
    
    def generate_vehicle(self, prompt: str, output_dir: str = None, output_filename: str = None, samples: int = 16) -> Tuple[Optional[Path], str]:
        """
        Generate a high-quality vehicle image based on the prompt.
        Returns: (image_path, vehicle_type)
        """
        vehicle_type = self._get_vehicle_type(prompt)
        classes, truncation, clip_prompt = self._get_optimal_params(vehicle_type)
        
        # Set random seed based on time for variety
        seed = int(time.time()) % 10000
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        # Fix directory handling
        if output_dir:
            # Convert to Path and ensure directory exists
            tmp_dir = Path(output_dir)
            os.makedirs(tmp_dir, exist_ok=True)
            if output_filename:
                img_path = tmp_dir / f"{output_filename}.png"
            else:
                img_path = tmp_dir / f"vehicle_{int(time.time())}.png"
        else:
            # Create temp directory
            tmp_dir = Path.cwd() / f"_vehicle_gen_{int(time.time())}"
            os.makedirs(tmp_dir, exist_ok=True)
            img_path = tmp_dir / f"vehicle_{int(time.time())}.png"
        
        # Try different models and truncations for best results
        best_image = None
        best_score = -1.0
        all_paths = []
        
        try:
            # Try BigGAN-deep-256 with different truncation values
            model_initialized = _init_biggan(self.device, "biggan-deep-256")
            if not model_initialized:
                print("Warning: Failed to initialize biggan-deep-256, trying fallback model")
                model_initialized = _init_biggan(self.device, "biggan-256")
                if not model_initialized:
                    print("Error: Failed to initialize BigGAN models")
                    return None, vehicle_type
            
            # Generate images
            for t in [truncation, truncation-0.05]:
                # Fix: Call BigGAN with existing directory
                try:
                    paths = _biggan_generate(classes, samples // 2, t, tmp_dir)
                    all_paths.extend(paths)
                    
                    # Get best by CLIP score
                    for p in paths:
                        score = _clip_score(p, clip_prompt)
                        if score > best_score:
                            best_score = score
                            best_image = p
                except Exception as e:
                    print(f"Error generating with truncation {t}: {e}")
            
            # If poor results, try with biggan-256
            if best_score < 0.22 and model_initialized:
                _init_biggan(self.device, "biggan-256")
                for t in [truncation, truncation-0.05]:
                    try:
                        paths = _biggan_generate(classes, samples // 2, t, tmp_dir)
                        all_paths.extend(paths)
                        
                        # Get best by CLIP score
                        for p in paths:
                            score = _clip_score(p, clip_prompt)
                            if score > best_score:
                                best_score = score
                                best_image = p
                    except Exception as e:
                        print(f"Error generating fallback with truncation {t}: {e}")
            
            # Save the best image to the expected path
            if best_image:
                result = Image.open(str(best_image)).convert("RGB")
                result.save(img_path)
                
                # Clean up temporary files except the final output
                for p in all_paths:
                    if str(p) != str(img_path):
                        try:
                            p.unlink()
                        except:
                            pass
                
                return img_path, vehicle_type
            
            return None, vehicle_type
            
        except Exception as e:
            import traceback
            print(f"Vehicle generation error: {e}")
            print(traceback.format_exc())
            return None, vehicle_type

# Singleton instance for reuse
_VEHICLE_GEN: Optional[VehicleGenerator] = None

def get_vehicle_generator(device: Optional[torch.device] = None) -> VehicleGenerator:
    """Get or create the vehicle generator singleton."""
    global _VEHICLE_GEN
    if _VEHICLE_GEN is None:
        _VEHICLE_GEN = VehicleGenerator(device)
    return _VEHICLE_GEN

def is_vehicle_prompt(prompt: str) -> bool:
    """Check if a prompt is specifically asking for a vehicle."""
    prompt_lower = prompt.lower().strip()
    
    # Check for direct vehicle type mentions
    for vehicle_type in VEHICLE_CLASS_MAP.keys():
        if vehicle_type in prompt_lower.split():
            return True
            
    # Common vehicle phrases
    vehicle_phrases = [
        "a car", "the car", "a truck", "the truck",
        "a motorcycle", "the motorcycle", "a bike", "the bike",
        "a bicycle", "the bicycle", "a bus", "the bus",
        "a vehicle", "the vehicle"
    ]
    
    for phrase in vehicle_phrases:
        if phrase in prompt_lower:
            return True
            
    return False

def generate_vehicle_image(prompt: str, output_dir: str = None, output_filename: str = None, device: Optional[torch.device] = None) -> Tuple[Optional[Path], str]:
    """
    Generate a vehicle image based on the prompt.
    Returns: (image_path, vehicle_type)
    """
    gen = get_vehicle_generator(device)
    return gen.generate_vehicle(prompt, output_dir, output_filename)
