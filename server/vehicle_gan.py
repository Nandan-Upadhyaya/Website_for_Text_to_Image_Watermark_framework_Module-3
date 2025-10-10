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
import re

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
    "vehicle": ["sports car", "convertible", "pickup", "motorcycle", "mountain bike"],

    # NEW: Animal mappings (ImageNet class names)
    "lion": ["lion"],
    "tiger": ["tiger"],
    "leopard": ["leopard", "jaguar", "cheetah"],
    "cheetah": ["cheetah"],
    "jaguar": ["jaguar"],
    "panther": ["leopard"],  # alias
    "wolf": ["wolf"],
    "fox": ["red fox"],
    "bear": ["brown bear"],
    "zebra": ["zebra"],
    "giraffe": ["giraffe"],
    "elephant": ["African elephant", "Indian elephant"],
    "rhinoceros": ["rhinoceros"],
    "rhino": ["rhinoceros"],  # alias
    "hippopotamus": ["hippopotamus"],
    "hippo": ["hippopotamus"],  # alias
    "crocodile": ["crocodile"],
    "alligator": ["American alligator"],
    "deer": ["deer"],
    "moose": ["moose"],
    "bison": ["bison"],
    "buffalo": ["water buffalo"],
    "horse": ["horse"],
    "cow": ["cow", "ox"],
    "goat": ["goat"],
    "sheep": ["ram", "bighorn"],
    "camel": ["Arabian camel"],
    "dog": ["Labrador retriever", "German shepherd", "Siberian husky"],
    "puppy": ["Labrador retriever"],  # alias
    "cat": ["tabby", "Persian cat", "Egyptian cat"],
    "kitten": ["tabby"],  # alias
    "panda": ["giant panda"],
    "koala": ["koala"],
    "kangaroo": ["kangaroo"],
    "otter": ["otter"],
    "raccoon": ["raccoon"],
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
    "vehicle": 0.32,  # generic fallback

    # NEW: Animal defaults
    "lion": 0.32,
    "tiger": 0.32,
    "leopard": 0.32,
    "cheetah": 0.32,
    "jaguar": 0.32,
    "panther": 0.32,
    "wolf": 0.34,
    "fox": 0.34,
    "bear": 0.34,
    "zebra": 0.35,
    "giraffe": 0.35,
    "elephant": 0.36,
    "rhinoceros": 0.36,
    "rhino": 0.36,
    "hippopotamus": 0.36,
    "hippo": 0.36,
    "crocodile": 0.36,
    "alligator": 0.36,
    "deer": 0.34,
    "moose": 0.36,
    "bison": 0.36,
    "buffalo": 0.36,
    "horse": 0.33,
    "cow": 0.35,
    "goat": 0.35,
    "sheep": 0.35,
    "camel": 0.36,
    "dog": 0.33,
    "puppy": 0.33,
    "cat": 0.33,
    "kitten": 0.33,
    "panda": 0.34,
    "koala": 0.34,
    "kangaroo": 0.35,
    "otter": 0.34,
    "raccoon": 0.34,
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
    "vehicle": "a photo of a vehicle, full view, realistic",

    # NEW: Animal prompts
    "lion": "a photo of a lion, full body, realistic, in the wild",
    "tiger": "a photo of a tiger, full body, realistic, in the jungle",
    "leopard": "a photo of a leopard, full body, realistic",
    "cheetah": "a photo of a cheetah, full body, realistic",
    "jaguar": "a photo of a jaguar, full body, realistic",
    "panther": "a photo of a black panther, full body, realistic",
    "wolf": "a photo of a wolf, full body, realistic",
    "fox": "a photo of a red fox, full body, realistic",
    "bear": "a photo of a brown bear, full body, realistic",
    "zebra": "a photo of a zebra, full body, realistic",
    "giraffe": "a photo of a giraffe, full body, realistic",
    "elephant": "a photo of an elephant, full body, realistic",
    "rhinoceros": "a photo of a rhinoceros, full body, realistic",
    "rhino": "a photo of a rhinoceros, full body, realistic",
    "hippopotamus": "a photo of a hippopotamus, full body, realistic",
    "hippo": "a photo of a hippopotamus, full body, realistic",
    "crocodile": "a photo of a crocodile, full body, realistic",
    "alligator": "a photo of an alligator, full body, realistic",
    "deer": "a photo of a deer, full body, realistic",
    "moose": "a photo of a moose, full body, realistic",
    "bison": "a photo of a bison, full body, realistic",
    "buffalo": "a photo of a water buffalo, full body, realistic",
    "horse": "a photo of a horse, full body, realistic",
    "cow": "a photo of a cow, full body, realistic",
    "goat": "a photo of a goat, full body, realistic",
    "sheep": "a photo of a sheep, full body, realistic",
    "camel": "a photo of a camel, full body, realistic",
    "dog": "a photo of a dog, full body, realistic",
    "puppy": "a photo of a puppy, full body, realistic",
    "cat": "a photo of a cat, full body, realistic",
    "kitten": "a photo of a kitten, full body, realistic",
    "panda": "a photo of a giant panda, full body, realistic",
    "koala": "a photo of a koala, full body, realistic",
    "kangaroo": "a photo of a kangaroo, full body, realistic",
    "otter": "a photo of an otter, full body, realistic",
    "raccoon": "a photo of a raccoon, full body, realistic",
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
        prompt_lower = (prompt or '').lower()
        # Strip punctuation and collapse whitespace so "A giraffe." -> "a giraffe"
        cleaned = re.sub(r'[^a-z0-9\s]+', ' ', prompt_lower)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        padded = f" {cleaned} "
        for vehicle_type in VEHICLE_CLASS_MAP.keys():
            if f" {vehicle_type} " in padded:
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
    """Check if a prompt is specifically asking for a vehicle or mapped animal (BigGAN path)."""
    prompt_lower = (prompt or '').lower()
    # Strip punctuation and collapse spaces: "A lion." -> "a lion"
    cleaned = re.sub(r'[^a-z0-9\s]+', ' ', prompt_lower)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    padded = f" {cleaned} "

    # Match any key (vehicles + animals), including multi-word keys
    for key in VEHICLE_CLASS_MAP.keys():
        if f" {key} " in padded:
            return True

    # Fallback common phrases (vehicles + animals)
    vehicle_phrases = [
        "a car", "the car", "a truck", "the truck",
        "a motorcycle", "the motorcycle", "a bike", "the bike",
        "a bicycle", "the bicycle", "a bus", "the bus",
        "a vehicle", "the vehicle",
        # animals (explicit phrase fallback)
        "a lion", "the lion", "a tiger", "the tiger", "a giraffe", "the giraffe",
        "a zebra", "the zebra", "an elephant", "the elephant", "a bear", "the bear",
        "a fox", "the fox", "a wolf", "the wolf", "a cow", "the cow", "a horse", "the horse"
    ]
    for phrase in vehicle_phrases:
        if f" {phrase} " in padded:
            return True

    return False

def generate_vehicle_image(prompt: str, output_dir: str = None, output_filename: str = None, device: Optional[torch.device] = None) -> Tuple[Optional[Path], str]:
    """
    Generate a vehicle image based on the prompt.
    Returns: (image_path, vehicle_type)
    """
    gen = get_vehicle_generator(device)
    return gen.generate_vehicle(prompt, output_dir, output_filename)
