# stylegan_wrapper.py
import torch
import os

import numpy as np
import PIL.Image
import pickle
import sys
from PIL import Image
from pathlib import Path

STYLEGAN2_ADA_PATH = Path(__file__).parent / "stylegan2-ada-pytorch"
sys.path.insert(0, str(STYLEGAN2_ADA_PATH))

STYLEGAN_MODELS = {
    "human": r"C:\Users\nanda\OneDrive\Desktop\WatermarkGAN\AI-Image-Suite\server\models\stylegan2_ffhq.pkl",
    "dog": r"C:\Users\nanda\OneDrive\Desktop\WatermarkGAN\AI-Image-Suite\server\models\stylegan2_afhqdog.pkl",
    "cat": r"C:\Users\nanda\OneDrive\Desktop\WatermarkGAN\AI-Image-Suite\server\models\stylegan2_afhqcat.pkl",
    "wild": r"C:\Users\nanda\OneDrive\Desktop\WatermarkGAN\AI-Image-Suite\server\models\stylegan2_afhqwild.pkl"
}

# BigGAN-backed subject generator shim to replace StyleGAN usage transparently.
import time
from pathlib import Path
from typing import List, Optional

# Optional CLIP for re-ranking
_CLIP = {"model": None, "preprocess": None, "device": None}

def _init_clip(device: torch.device) -> bool:
	# ...existing code...
	try:
		import clip  # type: ignore
		if _CLIP["model"] is None:
			_CLIP["model"], _CLIP["preprocess"] = clip.load("ViT-B/32", device=device)
			_CLIP["model"].eval()
			_CLIP["device"] = device
		return True
	except Exception:
		return False

def _clip_score(img_path: Path, text: str) -> float:
	try:
		import clip
		if _CLIP["model"] is None:
			return 0.0
		model = _CLIP["model"]; preprocess = _CLIP["preprocess"]; device = _CLIP["device"]
		im = preprocess(Image.open(str(img_path)).convert("RGB")).unsqueeze(0).to(device)
		tx = clip.tokenize([text]).to(device)
		with torch.no_grad():
			iv = model.encode_image(im); tv = model.encode_text(tx)
			iv = iv/iv.norm(dim=-1, keepdim=True); tv = tv/tv.norm(dim=-1, keepdim=True)
			return float((iv @ tv.T).squeeze().item())
	except Exception:
		return 0.0

# BigGAN context
_BG = {"model": None, "device": None}
# Keep fallback map but prefer dynamic resolution below
_IMAGENET_NAMES = {
    # ...existing animal mappings...
    "dog": ["Labrador retriever","Siberian husky","golden retriever","German shepherd"],
    "cat": ["tabby","tiger cat","Egyptian cat"],
    # Updated vehicle mappings based on successful debug results
    "car": ["convertible", "sports car", "racer"],  # These produced the good blue car images
    "sports car": ["sports car", "racer", "convertible"],
    "truck": ["pickup", "trailer truck"],
    "pickup": ["pickup"],
    "pickup truck": ["pickup"],
    "bus": ["minibus", "school bus"],
    "motorcycle": ["motorcycle", "motor scooter"],
    "bike": ["bicycle", "mountain bike"],
    "bicycle": ["bicycle", "mountain bike"],
    "scooter": ["motor scooter"],
}

# ---- NEW: robust ImageNet class resolution for BigGAN ----
try:
    # pytorch_pretrained_biggan exposes the canonical ImageNet class strings it knows
    from pytorch_pretrained_biggan.utils import IMAGENET_CLASSES as _IMAGENET_CLASS_STRS
    _IMAGENET_CLASS_STRS_LC = [s.lower() for s in _IMAGENET_CLASS_STRS]
except Exception:
    _IMAGENET_CLASS_STRS = []
    _IMAGENET_CLASS_STRS_LC = []

_VEHICLE_TERMS = {"car","sports car","truck","pickup","pickup truck","bus","motorcycle","bike","bicycle","scooter"}

# NEW: exclude part/close-up classes that cause abstract results
_EXCLUDE_PART_KEYWORDS = {
    "wheel","rim","brake","brakes","odometer","speedometer","dashboard","windscreen","windshield",
    "seat","headlight","taillight","grille","mirror","hood","bonnet","bumper","door","door handle",
    "tire","tyre","indicator","turn signal","fender","license","plate","knob","lever","steering","muffler"
}

def _find_classes_by_keywords(keywords: List[str]) -> List[str]:
    """Search BigGAN's ImageNet class strings by keywords; return deduped list preserving order."""
    if not _IMAGENET_CLASS_STRS:
        return []
    out: List[str] = []
    for kw in keywords:
        k = kw.strip().lower()
        if not k:
            continue
        for i, s in enumerate(_IMAGENET_CLASS_STRS_LC):
            if k in s:
                # filter out parts/close-ups
                if any(ex in s for ex in _EXCLUDE_PART_KEYWORDS):
                    continue
                out.append(_IMAGENET_CLASS_STRS[i])
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for s in out:
        if s not in seen:
            seen.add(s)
            deduped.append(s)
    return deduped

def _resolve_names(entity_type: str) -> List[str]:
    """Resolve user term to valid BigGAN ImageNet class strings."""
    key = (entity_type or "").strip().lower()
    if not key:
        return ["sports car"]  # safe default
    # vehicle-focused keyword expansion
    synonyms: dict[str, List[str]] = {
        "car": ["car", "sports car", "convertible", "limousine", "minivan", "cab", "taxi", "race car"],
        "sports car": ["sports car", "race car", "racer"],
        "truck": ["truck", "pickup", "trailer truck", "tow truck", "garbage truck"],
        "pickup": ["pickup", "pickup truck"],
        "pickup truck": ["pickup truck", "pickup"],
        "bus": ["bus", "minibus", "school bus"],
        "motorcycle": ["motorcycle", "bike", "motor scooter", "moped"],
        "bike": ["bicycle", "mountain bike", "bicycle-built-for-two"],
        "bicycle": ["bicycle", "mountain bike", "bicycle-built-for-two"],
        "scooter": ["motor scooter", "moped"],
    }
    # animals fallback synonyms (kept minimal here)
    if key not in synonyms:
        # Try dynamic search with the raw key first
        found = _find_classes_by_keywords([key])
        if found:
            return found
        # Fallback to static map if available
        return _IMAGENET_NAMES.get(key, [entity_type])
    # Vehicle case: search by expanded keywords
    keywords = synonyms[key]
    found = _find_classes_by_keywords(keywords)
    return found or _IMAGENET_NAMES.get(key, [entity_type])

def _init_biggan(device: torch.device, model_name: str = "biggan-deep-256") -> bool:
    try:
        from pytorch_pretrained_biggan import BigGAN  # type: ignore
        # NEW: reload if model name changed
        if _BG["model"] is None or getattr(_BG.get("model"), "model_name", "") != model_name:
            _BG["model"] = BigGAN.from_pretrained(model_name).to(device).eval()
            setattr(_BG["model"], "model_name", model_name)
            _BG["device"] = device
        return True
    except Exception:
        return False

def _biggan_generate(names: List[str], n: int, trunc: float, out_dir: Path) -> List[Path]:
    paths: List[Path] = []
    try:
        from pytorch_pretrained_biggan import truncated_noise_sample, one_hot_from_names
    except Exception as e:
        print(f"Error importing pytorch_pretrained_biggan: {e}")
        return paths
    
    # Ensure output directory exists
    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    device = _BG["device"] or torch.device("cpu")
    for i in range(n):
        try:
            # Pick a single class name per sample to keep batch size = 1
            name = names[i % len(names)] if len(names) > 0 else "sports car"
            z = torch.from_numpy(truncated_noise_sample(truncation=trunc, batch_size=1)).to(device)
            class_vec_np = one_hot_from_names([name], batch_size=1)  # shape (1, 1000)
            y = torch.from_numpy(class_vec_np).to(device)
            with torch.no_grad():
                img = _BG["model"](z, y, trunc)
            arr = (img.clamp(-1,1).add(1).div(2.0)*255).cpu().numpy().astype(np.uint8)[0]
            arr = np.transpose(arr, (1,2,0))  # HWC RGB
            
            # Generate output path and ensure directory exists
            p = out_dir / f"biggan_{int(time.time()*1000)}_{i}.png"
            
            # Save the image with better error handling
            pil_img = Image.fromarray(arr)
            pil_img.save(str(p))
            paths.append(p)
        except Exception as e:
            print(f"Error generating image {i}: {e}")
            continue
            
    return paths

def _rank_by_clip(paths: List[Path], text: str) -> Optional[Path]:
	if not paths:
		return None
	if _CLIP["model"] is None:
		return paths[0]
	best = paths[0]; best_s = -1.0
	for p in paths:
		s = _clip_score(p, text)
		if s > best_s:
			best_s, best = s, p
	return best

def _to_rgba_cutout(rgb_path: Path) -> Optional[Image.Image]:
	try:
		from rembg import remove
		rgb = Image.open(str(rgb_path)).convert("RGB")
		res = remove(np.array(rgb))
		if isinstance(res, bytes):
			from io import BytesIO
			return Image.open(BytesIO(res)).convert("RGBA")
		return Image.fromarray(res).convert("RGBA")
	except Exception:
		# fallback: no alpha
		return Image.open(str(rgb_path)).convert("RGBA")

# NEW: helper to generate N, rank by CLIP, and return (best_path, best_score)
def _gen_and_rank(names: List[str], trunc: float, n_samples: int, text: str, tmp: Path) -> tuple[Optional[Path], float, List[Path]]:
    paths = _biggan_generate(names, n_samples, trunc, tmp)
    if not paths:
        return None, -1.0, []
    best = _rank_by_clip(paths, text) or paths[0]
    # CLIP score if available
    try:
        score = _clip_score(best, text) if _CLIP["model"] is not None else 0.0
    except Exception:
        score = 0.0
    return best, float(score), paths

# NEW: import config flags for bypass
from config import SIMPLE_BIGGAN_ONLY, SIMPLE_BIGGAN_ONLY_SUBJECTS

# Ensure vehicle subjects are always included in the simple BigGAN-only set
VEHICLE_TERMS = {"car", "sports car", "truck", "pickup", "pickup truck", "bus", "motorcycle", "bike", "bicycle", "scooter"}
_SIMPLE_ONLY_SET = set(s.lower() for s in SIMPLE_BIGGAN_ONLY_SUBJECTS) | {"dog", "cat"} | VEHICLE_TERMS

class StyleGANGenerator:
    """Shim class used by app.py; internally uses BigGAN to return RGBA full-body subjects."""
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or (torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
        _init_biggan(self.device)
        _init_clip(self.device)  # optional

    def load_model(self, entity_type: str):
        # Kept for API compatibility; BigGAN is already initialized.
        return None

    def generate(self, entity_type: str, seed: int = 0, truncation: float = 0.5, samples: int = 8, prompt: Optional[str] = None) -> Image.Image:
        np.random.seed(seed); torch.manual_seed(seed
)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Resolve valid ImageNet class names and tune sampling for vehicles
        names = _resolve_names(entity_type)
        is_vehicle = entity_type.strip().lower() in _VEHICLE_TERMS
        
        # Using what we learned from the debug results - lower truncation works better for vehicles
        if is_vehicle:
            # Use these specific truncation values that produced good car results in debug
            trunc_list = [0.35, 0.3]  
            n_samples = 16  # Generate more samples for better selection
        else:
            trunc_list = [min(truncation, 0.42)]
            n_samples = max(samples, 8)

        tmp = Path.cwd() / f"_biggan_tmp_{int(time.time())}"
        tmp.mkdir(parents=True, exist_ok=True)
        
        # Better CLIP text for vehicles based on what worked in the debug output
        if is_vehicle:
            entity_key = entity_type.strip().lower()
            if entity_key == "car" or entity_key == "sports car":
                txt = prompt or "a photo of a blue convertible car, full vehicle, side view, clear"
            else:
                txt = prompt or f"a photo of a {entity_type}, full vehicle, side view"
        else:
            txt = prompt or f"a full body {entity_type}"

        best_overall, best_score = None, -1.0
        all_paths: List[Path] = []

        # Try biggan-deep-256 first
        if _init_biggan(self.device, "biggan-deep-256"):
            for t in trunc_list:
                b, s, paths = _gen_and_rank(names, t, n_samples, txt, tmp)
                all_paths.extend(paths)
                if s > best_score and b is not None:
                    best_overall, best_score = b, s

        # Fallback: try biggan-256 if vehicle and score still low
        if is_vehicle and best_score < 0.22:
            if _init_biggan(self.device, "biggan-256"):
                for t in trunc_list:
                    b, s, paths = _gen_and_rank(names, t, n_samples, txt, tmp)
                    all_paths.extend(paths)
                    if s > best_score and b is not None:
                        best_overall, best_score = b, s

        # If everything failed, raise
        if best_overall is None:
            # cleanup
            for p in all_paths:
                try: p.unlink()
                except: pass
            try: tmp.rmdir()
            except: pass
            raise RuntimeError("BigGAN generation failed. Install pytorch_pretrained_biggan.")

        # BigGAN-only bypass for simple subjects (includes vehicles)
        bypass = SIMPLE_BIGGAN_ONLY and (entity_type.lower() in _SIMPLE_ONLY_SET)
        if bypass:
            full_img = Image.open(str(best_overall)).convert("RGB")
            setattr(full_img, "_biggan_full_only", True)  # signal compositing to skip DF-GAN
            # cleanup temp images
            for p in all_paths:
                try: p.unlink()
                except: pass
            try: tmp.rmdir()
            except: pass
            return full_img

        # Default: return RGBA cutout for compositing
        rgba = _to_rgba_cutout(best_overall)
        for p in all_paths:
            try: p.unlink()
            except: pass
        try: tmp.rmdir()
        except: pass
        return rgba
