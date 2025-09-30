"""
Debug tool to analyze BigGAN vehicle generation.
Run with: python vehicle_debug.py car
"""
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, List, Tuple

# Import from stylegan_wrapper
sys.path.insert(0, os.path.dirname(__file__))
from stylegan_wrapper import (_init_biggan, _init_clip, _clip_score, 
                             _resolve_names, _find_classes_by_keywords,
                             _IMAGENET_CLASS_STRS, _biggan_generate)

# Important ImageNet vehicle classes that tend to work well
GOOD_VEHICLE_CLASSES = {
    "car": ["sports car", "convertible", "racer", "cab", "beach wagon", "station wagon"],
    "truck": ["pickup", "tractor", "trailer truck", "moving van"],
    "bus": ["minibus", "school bus", "trolleybus"],
    "motorcycle": ["moped", "motor scooter"],
    "bicycle": ["mountain bike", "tandem bicycle"]
}

def generate_debug_grid(entity: str, truncation: float = 0.4, samples: int = 16, 
                        save_path: Optional[str] = None):
    """Generate a grid of samples with their class names and CLIP scores."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize models
    _init_biggan(device, "biggan-deep-256")
    _init_clip(device)
    
    # Resolve classes and print them
    names = _resolve_names(entity)
    print(f"Resolved classes for '{entity}':")
    for name in names:
        print(f"  - {name}")
    
    # Force-add good vehicle classes if available
    if entity.lower() in GOOD_VEHICLE_CLASSES:
        hardcoded = GOOD_VEHICLE_CLASSES[entity.lower()]
        print(f"\nAdding hardcoded classes:")
        for h in hardcoded:
            if h not in names:
                print(f"  + {h}")
                names.append(h)

    # Generate temp dir
    tmp = Path.cwd() / f"_debug_biggan_{int(time.time())}"
    tmp.mkdir(parents=True, exist_ok=True)
    
    # Generate images
    paths = []
    for t in [truncation, truncation-0.05]:
        print(f"Generating with truncation={t:.2f}")
        paths.extend(_biggan_generate(names, samples, t, tmp))
    
    # Calculate CLIP scores
    target_text = f"a photo of a {entity}, full vehicle, centered"
    scores = []
    for p in paths:
        score = _clip_score(p, target_text)
        class_name = p.stem.split('_')[-2] if '_' in p.stem else "unknown"
        scores.append((p, score, class_name))
    
    # Sort by score
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Create grid
    if not scores:
        print("No images generated!")
        return
        
    # Grid size
    cols = min(4, len(scores))
    rows = (len(scores) + cols - 1) // cols
    
    # Load and resize images
    cell_size = 256
    grid_w = cols * cell_size
    grid_h = rows * cell_size
    
    grid = Image.new('RGB', (grid_w, grid_h), color='white')
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
        
    # Place images in grid with scores
    for i, (path, score, class_name) in enumerate(scores):
        row = i // cols
        col = i % cols
        img = Image.open(str(path)).resize((cell_size, cell_size), Image.LANCZOS)
        grid.paste(img, (col * cell_size, row * cell_size))
        
        # Add text
        draw = ImageDraw.Draw(grid)
        text = f"{class_name}\nCLIP: {score:.3f}"
        draw.text((col * cell_size + 5, row * cell_size + 5), text, fill="white", 
                  stroke_fill="black", stroke_width=2, font=font)
    
    # Save or show
    if save_path:
        grid.save(save_path)
        print(f"Grid saved to {save_path}")
    else:
        grid_path = str(tmp / f"{entity}_grid.jpg")
        grid.save(grid_path)
        print(f"Grid saved to {grid_path}")
    
    # Clean up
    for p in paths:
        try:
            p.unlink()
        except:
            pass
            
if __name__ == "__main__":
    entity = sys.argv[1] if len(sys.argv) > 1 else "car"
    generate_debug_grid(entity, truncation=0.4, samples=12, 
                      save_path=f"biggan_{entity}_samples.jpg")
    print(f"Run with different entities: python vehicle_debug.py [car|truck|motorcycle|etc]")
