import os
import sys
import time
import random
import numpy as np
import torch
from pathlib import Path
import nltk
from typing import List, Tuple, Optional  # <-- added here for py38-compatible typing

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import configuration
from config import DF_GAN_PATH, DF_GAN_CODE_PATH, DF_GAN_SRC_PATH

# Ensure DF-GAN code is in the path
sys.path.insert(0, str(DF_GAN_CODE_PATH))
sys.path.insert(0, str(DF_GAN_SRC_PATH))

# Import directly from the DF-GAN repo code
from lib.utils import mkdir_p, load_netG, truncated_noise
# NEW: use blending helpers
from blend_utils import poisson_blend_rgba, inpaint_rect

# ---- existing CLIP helpers (_init_clip, _clip_score) ----
_CLIP_CTX = {"model": None, "preprocess": None, "device": None}


def _init_clip(device: torch.device):
    """Lazy-init CLIP; return True if available."""
    try:
        import clip  # noqa: F401
        if _CLIP_CTX["model"] is None:
            _CLIP_CTX["model"], _CLIP_CTX["preprocess"] = clip.load("ViT-B/32", device=device)
            _CLIP_CTX["model"].eval()
            _CLIP_CTX["device"] = device
        return True
    except Exception:
        return False


def _clip_score(image_path: Path, prompt: str) -> float:
    """Return cosine similarity between image and prompt using CLIP, or 0.0 if unavailable."""
    try:
        import clip
        from PIL import Image
        if _CLIP_CTX["model"] is None:
            return 0.0
        model = _CLIP_CTX["model"]
        preprocess = _CLIP_CTX["preprocess"]
        device = _CLIP_CTX["device"]
        img = preprocess(Image.open(str(image_path)).convert("RGB")).unsqueeze(0).to(device)
        txt = clip.tokenize([prompt]).to(device)
        with torch.no_grad():
            img_f = model.encode_image(img)
            txt_f = model.encode_text(txt)
            img_f = img_f / img_f.norm(dim=-1, keepdim=True)
            txt_f = txt_f / txt_f.norm(dim=-1, keepdim=True)
            return float((img_f @ txt_f.T).squeeze().item())
    except Exception:
        return 0.0


# ---- NEW: Tokenization and prompt utilities ----
def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
        except Exception:
            pass


def _lower_words(s: str):
    try:
        from nltk.tokenize import word_tokenize
        return [w.lower() for w in word_tokenize(s)]
    except Exception:
        return s.lower().split()


def _remove_subject_terms(text: str, subject_terms: list) -> str:
    words = _lower_words(text)
    kept = [w for w in words if w.lower() not in subject_terms]
    return " ".join(kept) if kept else ""


def _suggest_subject_prompt(prompt: str, species_fallback: str = "animal") -> str:
    words = _lower_words(prompt)
    actions = [k for k in ("walking", "sleeping", "lying", "sitting", "running", "standing") if k in words]
    species = species_fallback
    for cand in ("dog", "cat", "horse", "cow", "sheep", "goat", "bird", "lion", "tiger", "elephant", "zebra", "giraffe", "monkey", "deer", "fox", "wolf", "rabbit", "pig"):
        if cand in words:
            species = cand
            break
    view = "full body, side view" if ("side" in words or "profile" in words) else "full body"
    action_str = actions[0] if actions else ""
    return f"a realistic {species}, {view} {action_str}".strip()


# ---- NEW: BigGAN subject provider (optional) ----
_BIGGAN_CTX = {"model": None, "device": None}
_IMAGENET_FALLBACK = {
    "dog": ["Labrador retriever", "Siberian husky", "golden retriever", "German shepherd"],
    "cat": ["tabby", "tiger cat", "Egyptian cat"],
    "horse": ["sorrel", "zebra"],  # zebra not a horse but offers full-body structure
    "cow": ["ox", "cow"],
    "sheep": ["ram"],
    "goat": ["Angora"],
    "bird": ["cock", "hen"],  # generic
    "rabbit": ["hare"],
    "fox": ["red fox"],
    "wolf": ["timber wolf"],
    "elephant": ["African elephant"],
    "lion": ["lion"],
    "tiger": ["tiger"],
    "zebra": ["zebra"],
    "giraffe": ["giraffe"],
    "pig": ["hog"],
    "deer": ["elk"],
}


def _init_biggan(device: torch.device) -> bool:
    try:
        from pytorch_pretrained_biggan import BigGAN  # type: ignore
        if _BIGGAN_CTX["model"] is None:
            from config import SUBJECT_PROVIDER
            _BIGGAN_CTX["model"] = BigGAN.from_pretrained(SUBJECT_PROVIDER.get("biggan_model", "biggan-deep-256"))
            _BIGGAN_CTX["model"].to(device)
            _BIGGAN_CTX["model"].eval()
            _BIGGAN_CTX["device"] = device
        return True
    except Exception:
        return False


def _biggan_generate_set(names: list, n: int, truncation: float, out_dir: Path) -> List[Path]:  # <-- return type List[Path]
    try:
        from pytorch_pretrained_biggan import truncated_noise_sample, one_hot_from_names  # type: ignore
        from PIL import Image
    except Exception:
        return []
    model = _BIGGAN_CTX["model"]
    device = _BIGGAN_CTX["device"] or torch.device("cpu")
    paths = []
    for i in range(n):
        noise = truncated_noise_sample(truncation=truncation, batch_size=1)
        class_vec = one_hot_from_names(names, batch_size=1)
        noise = torch.from_numpy(noise).to(device)
        class_vec = torch.from_numpy(class_vec).to(device)
        with torch.no_grad():
            out = model(noise, class_vec, truncation)
        img = (out.clamp(-1, 1).add(1).div(2.0) * 255).cpu().numpy().astype(np.uint8)[0]
        img = np.transpose(img, (1, 2, 0))  # HWC RGB
        p = out_dir / f"biggan_{int(time.time()*1000)}_{i}.png"
        Image.fromarray(img).save(str(p))
        paths.append(p)
    return paths


def _biggan_subject(subject_prompt: str, work_dir: Path) -> Optional[Path]:  # <-- Path | None -> Optional[Path]
    from config import SUBJECT_PROVIDER
    if not _init_biggan(_CLIP_CTX["device"] or torch.device("cpu")):
        return None
    # pick candidate class names
    words = set(_lower_words(subject_prompt))
    species = "dog"
    for k in _IMAGENET_FALLBACK.keys():
        if k in words:
            species = k
            break
    names = _IMAGENET_FALLBACK.get(species, [species])
    # generate many, then CLIP re-rank
    n = max(1, SUBJECT_PROVIDER.get("biggan_samples", 8))
    trunc = float(SUBJECT_PROVIDER.get("biggan_truncation", 0.5))
    out_dir = work_dir / "biggan_tmp"
    mkdir_p(out_dir)
    paths = _biggan_generate_set(names, n, trunc, out_dir)
    if not paths:
        return None
    _init_clip(_CLIP_CTX["device"] or torch.device("cpu"))
    best, _ = _clip_rank(paths, subject_prompt)
    if best is None:
        return None
    # alpha-matte via rembg
    try:
        from rembg import remove
        from PIL import Image
        rgb = Image.open(str(best)).convert("RGB")
        cut = remove(np.array(rgb))
        # rembg may return RGBA array or bytes
        if isinstance(cut, bytes):
            from io import BytesIO
            cut_img = Image.open(BytesIO(cut)).convert("RGBA")
        else:
            cut_img = Image.fromarray(cut).convert("RGBA")
        out_path = work_dir / f"subject_biggan_{int(time.time())}.png"
        cut_img.save(str(out_path))
        return out_path
    except Exception:
        return None


# ---- NEW: Subject provider + compositing helpers ----
from PIL import Image, ImageFilter, ImageEnhance
import io
import subprocess
import shlex
import glob


def _clip_rank(paths: List[Path], prompt: str) -> Tuple[Optional[Path], float]:
    """Return best path by CLIP score, or (first, 0.0) if CLIP unavailable."""
    if not paths:
        return None, 0.0
    best = paths[0]
    best_s = 0.0
    if _CLIP_CTX["model"] is None:
        return best, best_s
    for p in paths:
        s = _clip_score(p, prompt)
        if s > best_s:
            best, best_s = p, s
    return best, best_s


def _list_images_recursive(root: Path) -> List[Path]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    files = []
    for pat in exts:
        files.extend([Path(p) for p in glob.glob(str(root / "**" / pat), recursive=True)])
    return files


def _find_asset_subject(subject_prompt: str) -> Optional[Path]:
    """Pick a subject RGBA cutout from assets dir; use CLIP to re-rank if available."""
    from config import SUBJECT_PROVIDER
    assets_dir = Path(SUBJECT_PROVIDER["assets_dir"])
    if SUBJECT_PROVIDER["mode"] != "assets" or not assets_dir.exists():
        return None
    candidates = _list_images_recursive(assets_dir)
    # quick filename filter on species keywords
    wl = ["dog", "cat", "horse", "cow", "sheep", "goat", "bird", "lion", "tiger", "elephant", "zebra", "giraffe", "monkey", "deer", "fox", "wolf", "rabbit", "pig"]
    words = set(_lower_words(subject_prompt))
    species = [w for w in wl if w in words]
    if species:
        s = species[0]
        candidates = [p for p in candidates if s in p.stem.lower() or s in str(p.parent).lower()]
    if not candidates:
        return None
    _init_clip(_CLIP_CTX["device"] or torch.device("cpu"))
    best, _ = _clip_rank(candidates, subject_prompt)
    return best


def _call_http_subject(subject_prompt: str, out_path: Path) -> Optional[Path]:
    """POST to external generator; expects it to save RGBA PNG at out_path."""
    from config import SUBJECT_PROVIDER
    try:
        import requests
    except Exception:
        return None
    endpoint = SUBJECT_PROVIDER["http_endpoint"]
    if SUBJECT_PROVIDER["mode"] != "http" or not endpoint:
        return None
    try:
        resp = requests.post(endpoint, json={"prompt": subject_prompt, "out": str(out_path)}, timeout=SUBJECT_PROVIDER["timeout_sec"])
        if resp.status_code == 200 and out_path.exists():
            return out_path
    except Exception:
        pass
    return None


def _call_cmd_subject(subject_prompt: str, out_path: Path) -> Optional[Path]:
    """Invoke an external command to synthesize the subject RGBA."""
    from config import SUBJECT_PROVIDER
    tmpl = SUBJECT_PROVIDER["cmd_template"]
    if SUBJECT_PROVIDER["mode"] != "cmd" or not tmpl:
        return None
    cmd = tmpl.replace("{prompt}", subject_prompt).replace("{out}", str(out_path))
    try:
        subprocess.run(shlex.split(cmd), check=True, timeout=SUBJECT_PROVIDER["timeout_sec"])
        return out_path if out_path.exists() else None
    except Exception:
        return None


def _ensure_rgba(img: Image.Image) -> Image.Image:
    return img.convert("RGBA") if img.mode != "RGBA" else img


def _alpha_from_subject(img: Image.Image) -> Optional[Image.Image]:
    """Return alpha mask; try rembg if alpha missing."""
    if img.mode == "RGBA":
        return img.split()[-1]
    # optional: rembg
    try:
        from rembg import remove
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        cut = remove(buf.getvalue())
        cut_img = Image.open(io.BytesIO(cut)).convert("RGBA")
        return cut_img.split()[-1]
    except Exception:
        return None


def _simple_color_match(subject_rgb: Image.Image, bg_rgb: Image.Image) -> Image.Image:
    """Match mean brightness roughly."""
    try:
        import numpy as np
        s = np.asarray(subject_rgb).astype(np.float32)
        b = np.asarray(bg_rgb).astype(np.float32)
        s_mean = s.mean(axis=(0, 1))
        b_mean = b.mean(axis=(0, 1))
        scale = (b_mean.mean() + 1e-6) / (s_mean.mean() + 1e-6)
        s2 = np.clip(s * scale, 0, 255).astype(np.uint8)
        return Image.fromarray(s2)
    except Exception:
        # fallback: PIL enhance
        enhancer = ImageEnhance.Brightness(subject_rgb)
        return enhancer.enhance(1.05)


def _composite_subject_on_bg(bg_path: Path, subject_path: Path, out_path: Optional[Path] = None) -> Optional[Path]:
    """Composite subject cutout onto background with soft shadow, optional inpaint, or Poisson blend."""
    from config import COMPOSITION_DEFAULTS as CD
    try:
        from PIL import Image
        bg = Image.open(str(bg_path)).convert("RGB")
        subj = Image.open(str(subject_path))
        subj = _ensure_rgba(subj)
        alpha = _alpha_from_subject(subj)
        if alpha is None:
            return None
        # scale subject to target height
        bw, bh = bg.size
        sw, sh = subj.size
        target_h = max(1, int(bh * CD["subject_height_frac"]))
        scale = target_h / sh
        subj_resized = subj.resize((max(1, int(sw * scale)), target_h), Image.LANCZOS)
        alpha_resized = subj_resized.split()[-1]
        subj_rgb = subj_resized.convert("RGB")

        # placement
        margin = int(bh * CD["bottom_margin_frac"])
        x = (bw - subj_resized.width) // 2
        y = bh - subj_resized.height - margin
        x = max(0, min(x, bw - subj_resized.width))
        y = max(0, min(y, bh - subj_resized.height))

        # Optional Poisson blending with optional inpaint under region
        if CD.get("use_poisson", True):
            # Prepare BGR images
            import cv2
            bg_bgr = cv2.cvtColor(np.array(bg), cv2.COLOR_RGB2BGR)
            fg_rgba = np.dstack((np.array(subj_rgb), np.array(alpha_resized)))
            if CD.get("inpaint_under_subject", True):
                rect = (x, y, subj_resized.width, subj_resized.height)
                bg_bgr = inpaint_rect(bg_bgr, rect, method="telea")
            blended_bgr = poisson_blend_rgba(bg_bgr, fg_rgba, position=(x, y), mixed=False)
            blended_rgb = cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
            from PIL import Image as _Image
            comp_img = _Image.fromarray(blended_rgb)
        else:
            # Fallback: simple alpha composite with optional color match
            from config import COMPOSITION_DEFAULTS as CD2
            if CD2["color_match"]:
                subj_rgb = _simple_color_match(subj_rgb, bg)
            comp = bg.convert("RGBA")
            comp.paste(subj_rgb, (x, y), alpha_resized)
            comp_img = comp.convert("RGB")

        if out_path is None:
            out_path = bg_path.with_name(bg_path.stem + "_composite.png")
        comp_img.save(str(out_path))
        return out_path
    except Exception:
        return None


class DFGANGenerator:
    """Wrapper class for DF-GAN text-to-image generation."""

    def __init__(self, model_path, data_dir, batch_size=1, use_cuda=True, seed=100, steps=50, guidance=7.5):
        self.model_path = model_path
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.seed = seed
        self.steps = steps
        self.guidance = guidance
        self.wordtoix = None
        self.text_encoder = None
        self.netG = None
        self.args = self._setup_args()

        # Set random seeds for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.use_cuda:
            torch.cuda.manual_seed_all(self.seed)
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.args.device = self.device
        self._load_models()

    def _setup_args(self):
        """Set up arguments for DF-GAN model."""
        from types import SimpleNamespace

        # Basic configuration based on the DF-GAN repo
        args = SimpleNamespace()
        args.z_dim = 100
        args.imsize = 256
        args.cuda = self.use_cuda
        args.manual_seed = self.seed
        args.multi_gpus = False
        args.imgs_per_sent = self.batch_size
        args.train = False
        args.truncation = True
        args.trunc_rate = 0.8
        args.checkpoint = self.model_path
        args.data_dir = self.data_dir
        args.samples_save_dir = str(Path.cwd() / "temp_output")

        # Required attributes based on the actual DF-GAN codebase
        args.local_rank = 0
        args.gpu_id = 0
        args.distributed = False
        args.cond_dim = 256
        args.batch_size = self.batch_size
        args.workers = 4
        args.gan_type = 'DFGAN'

        # Generator and Discriminator architecture configs from DF-GAN repo
        args.nf = 32  # Number of features
        args.gf_dim = 32
        args.df_dim = 64
        args.ef_dim = 256
        args.n_units = 32
        args.ch_size = 3   # Channel size - required for model initialization

        # TEXT namespace as configured in the actual repo
        args.TEXT = SimpleNamespace()
        args.TEXT.WORDS_NUM = 18
        args.TEXT.EMBEDDING_DIM = 256
        args.TEXT.CAPTIONS_PER_IMAGE = 10
        args.TEXT.HIDDEN_DIM = 128
        args.TEXT.RNN_TYPE = 'LSTM'

        if "CUB" in str(self.model_path):
            args.dataset = "birds"
            args.n_classes = 10
            args.encoder_epoch = 600
            args.encoder_path = str(DF_GAN_PATH / 'data' / 'birds' / 'DAMSMencoder')
        else:
            args.dataset = "coco"
            args.n_classes = 80
            args.encoder_epoch = 100
            args.encoder_path = str(DF_GAN_PATH / 'data' / 'coco' / 'DAMSMencoder')

        # Add additional required params based on repo code
        args.NET_G = ''
        args.NET_D = ''
        args.NET_E = ''
        args.WORKERS = 4  # Uppercase version also used
        args.B_VALIDATION = False
        args.stamp = 'default'

        return args

    def _load_models(self):
        """Load the DF-GAN models."""
        try:
            # Load word dictionary - Check multiple possible paths
            pickle_paths = [
                os.path.join(self.args.data_dir, f'captions_DAMSM.pickle'),
                os.path.join(self.args.data_dir, f'captions.pickle'),
                # Try with dataset folder name variations
                str(Path(self.args.data_dir).parent / "birds" / "captions_DAMSM.pickle"),
                str(Path(self.args.data_dir).parent / "bird" / "captions_DAMSM.pickle"),
                str(Path(self.args.data_dir).parent / "birds" / "captions.pickle"),
                str(Path(self.args.data_dir).parent / "bird" / "captions.pickle"),
                # Use config-based path instead of hardcoded path
                str(DF_GAN_PATH / 'data' / 'birds' / 'captions_DAMSM.pickle'),
                str(DF_GAN_PATH / 'data' / 'coco' / 'captions_DAMSM.pickle')
            ]

            pickle_path = None
            for path in pickle_paths:
                print(f"Checking for pickle at: {path}")
                if os.path.exists(path):
                    pickle_path = path
                    print(f"Found pickle file at: {pickle_path}")
                    break

            if pickle_path is None:
                raise FileNotFoundError(f"Cannot find pickle file in any of the expected locations: {pickle_paths}")

            # Load the pickle file
            import pickle
            with open(pickle_path, 'rb') as f:
                x = pickle.load(f)
                self.wordtoix = x[3]
                self.args.vocab_size = len(self.wordtoix)
                print(f"Loaded vocabulary with {self.args.vocab_size} words")

            # Load models using the exact code structure from the DF-GAN repo
            from lib.perpare import prepare_models

            # Check for encoder files and adjust encoder_epoch if needed
            encoder_path = Path(self.args.encoder_path)
            img_encoder = encoder_path / f"image_encoder{self.args.encoder_epoch}.pth"
            text_encoder = encoder_path / f"text_encoder{self.args.encoder_epoch}.pth"

            print(f"Loading image encoder from: {img_encoder}")
            print(f"Loading text encoder from: {text_encoder}")

            if not img_encoder.exists() or not text_encoder.exists():
                print(f"Warning: {img_encoder} not found. Trying to use image_encoder100.pth instead.")
                print(f"Warning: {text_encoder} not found. Trying to use text_encoder100.pth instead.")
                self.args.encoder_epoch = 100

                img_encoder = encoder_path / "image_encoder100.pth"
                text_encoder = encoder_path / "text_encoder100.pth"

                print(f"Image encoder path: {img_encoder}")
                print(f"Image encoder exists: {img_encoder.exists()}")
                print(f"Text encoder path: {text_encoder}")
                print(f"Text encoder exists: {text_encoder.exists()}")

            _, self.text_encoder, self.netG, _, _ = prepare_models(self.args)
            self.netG = load_netG(self.netG, str(self.model_path), False, train=False)
            self.netG.eval()
            print(f"Models loaded successfully from {self.model_path}")

        except Exception as e:
            print(f"Error loading models: {e}")
            raise

    def tokenize_text(self, prompt):
        """Convert a prompt to token indices based on the DF-GAN repo implementation"""
        # NEW: make sure punkt is present
        _ensure_nltk()
        # Convert text to token indices using the loaded wordtoix
        cap_len = len(prompt.split())
        tokens = nltk.tokenize.word_tokenize(prompt.lower())
        tokens = tokens[:18]

        cap = []
        for token in tokens:
            if token in self.wordtoix:
                cap.append(self.wordtoix[token])
            else:
                cap.append(0)  # Use 0 for unknown words

        # Pad to fixed length
        if len(cap) < 18:
            cap = cap + [0] * (18 - len(cap))

        cap = torch.tensor([cap], dtype=torch.long).to(self.device)
        cap_len = torch.tensor([min(cap_len, 18)], dtype=torch.long).to(self.device)

        return cap, cap_len

    def generate_image(self, prompt, output_dir=None):
        """Generate images based on text prompt."""
        if output_dir is None:
            output_dir = Path(f"temp_output_{int(time.time())}")
        else:
            output_dir = Path(output_dir)

        mkdir_p(output_dir)

        try:
            # Process the prompt as in the actual DF-GAN code
            cap, cap_len = self.tokenize_text(prompt)

            # Get text embeddings using the encoder
            with torch.no_grad():
                hidden = self.text_encoder.init_hidden(1)
                words_embs, sent_emb = self.text_encoder(cap, cap_len, hidden)
                sent_emb = sent_emb.detach()

            # Generate noise vector
            if self.args.truncation:
                noise = truncated_noise(self.batch_size, self.args.z_dim, self.args.trunc_rate)
                noise = torch.tensor(noise, dtype=torch.float).to(self.device)
            else:
                noise = torch.randn(self.batch_size, self.args.z_dim).to(self.device)

            # Generate image
            with torch.no_grad():
                fake_imgs = self.netG(noise, sent_emb)

                # Save image using torchvision
                import torchvision.utils as vutils
                img_path = output_dir / f"generated_{int(time.time())}.png"
                # Use value_range instead of range for newer PyTorch versions
                vutils.save_image(fake_imgs.data, str(img_path), nrow=1, normalize=True, value_range=(-1, 1))

            return img_path

        except Exception as e:
            print(f"Error generating image: {e}")
            import traceback
            print(traceback.format_exc())
            raise

    # ---- NEW: routing helpers ----
    def generate_images_best_of(self, prompt, best_of=4, out_dir=None):
        """Generate multiple DF-GAN images for a prompt and return list of paths."""
        paths = []
        out_dir = Path(out_dir) if out_dir else Path(f"temp_output_{int(time.time())}")
        mkdir_p(out_dir)
        base_seed = self.seed
        for i in range(best_of):
            self.seed = base_seed + i
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if self.use_cuda:
                torch.cuda.manual_seed_all(self.seed)
            p = self.generate_image(prompt, output_dir=out_dir)
            paths.append(p)
        self.seed = base_seed
        return paths

    def _make_background_prompt(self, prompt: str) -> str:
        """Remove subject terms and add negative constraints for clean background."""
        from config import SUBJECT_TERMS, NEGATIVE_SUBJECT_HINT
        cleaned = _remove_subject_terms(prompt, SUBJECT_TERMS)
        # ensure scene nouns persist; if emptied, fallback to common road/street templates
        if not cleaned.strip():
            # try infer scene keyword
            words = _lower_words(prompt)
            scene = "street" if "street" in words else "road" if "road" in words else "scene"
            cleaned = f"an empty {scene}"
        # append negative hints
        if NEGATIVE_SUBJECT_HINT not in cleaned:
            cleaned = f"{cleaned}, {NEGATIVE_SUBJECT_HINT}"
        return cleaned

    def generate_or_route(self, prompt: str, best_of=None, clip_threshold=None, use_clip=None):
        """Auto pipeline:
        - Generate best-of DF-GAN images
        - If best passes CLIP threshold -> return as ok
        - Else generate background-only and return needs_subject_composite with a suggested subject prompt
        """
        from config import ROUTER_DEFAULTS
        best_of = best_of if best_of is not None else ROUTER_DEFAULTS["best_of"]
        clip_threshold = clip_threshold if clip_threshold is not None else ROUTER_DEFAULTS["clip_threshold"]
        use_clip = ROUTER_DEFAULTS["use_clip"] if use_clip is None else use_clip

        if use_clip:
            _init_clip(self.device)

        candidates = self.generate_images_best_of(prompt, best_of=best_of)
        best_path = candidates[0]
        best_score = 0.0

        if use_clip and _CLIP_CTX["model"] is not None:
            for p in candidates:
                s = _clip_score(p, prompt)
                if s > best_score:
                    best_score, best_path = s, p

        if (use_clip and best_score >= clip_threshold) or (not use_clip):
            return {
                "status": "ok",
                "image": str(best_path),
                "router": {"clip_score": best_score, "clip_threshold": clip_threshold, "used_clip": bool(_CLIP_CTX["model"] is not None)},
            }

        bg_prompt = self._make_background_prompt(prompt)
        bg_path = self.generate_image(bg_prompt)
        suggested_subject = _suggest_subject_prompt(prompt)
        return {
            "status": "needs_subject_composite",
            "background": str(bg_path),
            "background_prompt": bg_prompt,
            "suggested_subject_prompt": suggested_subject,
            "router": {"clip_score": best_score, "clip_threshold": clip_threshold, "used_clip": bool(_CLIP_CTX["model"] is not None)},
        }

    # ---- NEW: end-to-end pipeline ----
    def _get_subject_cutout(self, subject_prompt: str, work_dir: Path) -> Optional[Path]:
        """Try assets → biggan → http → cmd to obtain an RGBA subject cutout."""
        out_path = work_dir / f"subject_{int(time.time())}.png"
        # assets
        p = _find_asset_subject(subject_prompt)
        if p:
            return p
        # biggan (always attempt if available so we don't require StyleGAN weights)
        from config import SUBJECT_PROVIDER
        try_biggan = True  # force BigGAN attempt when assets missing
        if try_biggan and _init_biggan(_CLIP_CTX["device"] or torch.device("cpu")):
            p = _biggan_subject(subject_prompt, work_dir)
            if p:
                return p
        # http
        p = _call_http_subject(subject_prompt, out_path)
        if p:
            return p
        # cmd
        p = _call_cmd_subject(subject_prompt, out_path)
        if p:
            return p
        return None

    def generate_full(self, prompt: str, best_of: Optional[int] = None, clip_threshold: Optional[float] = None, use_clip: Optional[bool] = None, work_dir: Optional[Path] = None):
        """Full pipeline:
           - DF-GAN best-of + CLIP routing
           - If ok: return DF-GAN image
           - Else: background-only, subject cutout (BigGAN/assets/http/cmd), Poisson blend, return final
        """
        from config import SUBJECT_PROVIDER
        work_dir = Path(work_dir) if work_dir else Path(f"temp_pipeline_{int(time.time())}")
        mkdir_p(work_dir)
        route = self.generate_or_route(prompt, best_of=best_of, clip_threshold=clip_threshold, use_clip=use_clip)

        if route.get("status") == "ok":
            return {"status": "ok", "image": route["image"], "router": route.get("router", {})}

        # background and subject prompt
        bg_path = Path(route["background"])
        subj_prompt = route["suggested_subject_prompt"]

        # get subject cutout
        subj_cutout = self._get_subject_cutout(subj_prompt, work_dir)
        if not subj_cutout:
            return {
                "status": "needs_subject_assets",
                "background": str(bg_path),
                "background_prompt": route.get("background_prompt"),
                "suggested_subject_prompt": subj_prompt,
                "provider_mode": SUBJECT_PROVIDER["mode"],
                "hint": "Provide RGBA cutouts in SUBJECT_ASSETS_DIR or configure HTTP/CMD provider.",
            }

        # composite
        out_path = bg_path.with_name(bg_path.stem + "_final.png")
        comp = _composite_subject_on_bg(bg_path, subj_cutout, out_path=out_path)
        if not comp:
            return {
                "status": "compose_failed",
                "background": str(bg_path),
                "subject": str(subj_cutout),
                "hint": "Subject must have alpha channel or install rembg for auto-matting.",
            }

        return {
            "status": "completed",
            "image": str(comp),
            "background": str(bg_path),
            "subject": str(subj_cutout),
            "router": route.get("router", {}),
        }


# Helper functions for direct usage (ensure no duplicates)
def setup_generator(model_path, data_dir, use_cuda=True, seed=100):
    """Helper function to set up a generator instance."""
    return DFGANGenerator(model_path, data_dir, use_cuda=use_cuda, seed=seed)


def generate_image(generator, prompt, output_dir=None):
    """Helper function to generate an image with the given generator."""
    return generator.generate_image(prompt, output_dir)


def generate_or_route(generator: DFGANGenerator, prompt: str, best_of=None, clip_threshold=None, use_clip=None):
    """Auto router wrapper."""
    return generator.generate_or_route(prompt, best_of=best_of, clip_threshold=clip_threshold, use_clip=use_clip)


# NEW: one-call full pipeline
def generate_full(generator: DFGANGenerator, prompt: str, best_of=None, clip_threshold=None, use_clip=None, work_dir=None):
    return generator.generate_full(prompt, best_of=best_of, clip_threshold=clip_threshold, use_clip=use_clip, work_dir=work_dir)
# NEW: one-call full pipeline
def generate_full(generator: DFGANGenerator, prompt: str, best_of=None, clip_threshold=None, use_clip=None, work_dir=None):
    return generator.generate_full(prompt, best_of=best_of, clip_threshold=clip_threshold, use_clip=use_clip, work_dir=work_dir)
