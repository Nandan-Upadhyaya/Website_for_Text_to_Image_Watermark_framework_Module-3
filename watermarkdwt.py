import numpy as np
import pywt
import os
from PIL import Image, ImageDraw, ImageFont
from scipy.fftpack import dct, idct
from scipy.ndimage import gaussian_filter
from skimage.metrics import peak_signal_noise_ratio as PSNR
import cv2

# --- CONFIGURATION ---
IMAGE_HOST = r"D:\images2.png"  # Host image path
IMAGE_WATERMARK = r'D:\images.png'  # Watermark image path (only for mode='image')
WATERMARK_TEXT = "Nandan Upadhyaya"  # Text watermark (only for mode='text')
WATERMARK_MODE = 'image'  # 'image' or 'text'

HOST_SIZE = 512
WATERMARK_SIZE = 64  # CHANGED: Increased watermark size to 64x64
MODEL = 'haar'
LEVEL = 1  # CHANGED: Level 1 instead of 2 for better capacity
ALPHA = 0.28  # slightly stronger for robustness
REDUNDANCY = 8  # repeat each bit across more blocks for stronger majority vote

os.makedirs('./result', exist_ok=True)
os.makedirs('./attacks', exist_ok=True)

# =========================================================================
# === HELPER FUNCTIONS
# =========================================================================

def rgb_to_ycbcr(img_rgb):
    """Convert RGB to YCbCr"""
    ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    return ycbcr[:,:,0].astype(np.float64), ycbcr[:,:,1], ycbcr[:,:,2]

def ycbcr_to_rgb(Y, Cr, Cb):
    """Convert YCbCr back to RGB"""
    Y_uint = np.clip(Y, 0, 255).astype(np.uint8)
    ycbcr = np.stack([Y_uint, Cr, Cb], axis=2)
    return cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2RGB)

def load_image(path, size):
    """Load image as RGB numpy array"""
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return np.array(img, dtype=np.uint8)

def load_watermark_image(path, size):
    """Load watermark image as grayscale"""
    img = Image.open(path).convert('L')
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return np.array(img, dtype=np.float64)

def create_text_watermark(text, size, font_size=None):
    """Create text watermark as grayscale numpy array"""
    if font_size is None:
        font_size = max(12, size // 8)
    
    # Create image
    img = Image.new('L', (size, size), color=0)
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Measure text
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except:
        text_w, text_h = draw.textsize(text, font=font)
    
    # Center text
    x = (size - text_w) // 2
    y = (size - text_h) // 2
    
    draw.text((x, y), text, fill=255, font=font)
    return np.array(img, dtype=np.float64)

def apply_dct(image_array):
    """Apply 8x8 block-wise DCT"""
    h, w = image_array.shape
    h = (h // 8) * 8
    w = (w // 8) * 8
    image_array = image_array[:h, :w]
    
    all_subdct = np.zeros_like(image_array)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = image_array[i:i+8, j:j+8]
            subdct = dct(dct(block.T, norm="ortho").T, norm="ortho")
            all_subdct[i:i+8, j:j+8] = subdct
    return all_subdct

def inverse_dct(dct_array):
    """Apply 8x8 block-wise inverse DCT"""
    h, w = dct_array.shape
    all_subidct = np.zeros_like(dct_array)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = dct_array[i:i+8, j:j+8]
            subidct = idct(idct(block.T, norm="ortho").T, norm="ortho")
            all_subidct[i:i+8, j:j+8] = subidct
    return all_subidct

def _prepare_capacity_and_wm(Y_channel, wm_gray):
    """
    Compute LL capacity and resize -> binarize watermark to fit capacity with redundancy.
    Returns: ref_map {-1,+1} of shape (s,s), s, (h,w)
    """
    coeffs = pywt.wavedec2(Y_channel, MODEL, level=LEVEL)
    LL = coeffs[0]
    h, w = LL.shape
    blocks_h, blocks_w = (h // 8), (w // 8)
    cap_blocks = blocks_h * blocks_w
    # capacity with redundancy:
    s = int(np.floor(np.sqrt(max(1, cap_blocks // REDUNDANCY))))
    if s < 8:
        raise ValueError(f"Insufficient capacity in LL subband: {cap_blocks} blocks (need >= 8x8 with redundancy)")
    # Resize watermark to s x s
    wm_img = Image.fromarray(np.uint8(np.clip(wm_gray, 0, 255)))
    wm_resized = wm_img.resize((s, s), Image.Resampling.LANCZOS)
    wm_arr = np.array(wm_resized, dtype=np.float64)
    # Binarize using adaptive threshold (median is robust)
    thr = float(np.median(wm_arr))
    bits01 = (wm_arr >= thr).astype(np.uint8)
    ref_map = (bits01 * 2 - 1).astype(np.float64)  # {-1, +1}
    return ref_map, s, (h, w)

def _coeff_strength_from_ll(dct_ll):
    """
    Estimate stable per-image strength scale from median |(3,4) and (4,3)| coeffs.
    """
    a = np.abs(dct_ll[3::8, 4::8]).flatten()
    b = np.abs(dct_ll[4::8, 3::8]).flatten()
    med = np.median(np.concatenate([a, b])) + 1e-6
    return med

def _embed_pair_margin(block, bit, margin):
    """
    Enforce pairwise inequality between (3,4) and (4,3) by a margin.
    bit = +1 => c1 - c2 >= margin
    bit = -1 => c2 - c1 >= margin
    Adjust both coefficients minimally to satisfy margin.
    """
    c1 = block[3, 4]
    c2 = block[4, 3]
    diff = c1 - c2
    if bit > 0:
        if diff < margin:
            delta = 0.5 * (margin - diff)
            block[3, 4] = c1 + delta
            block[4, 3] = c2 - delta
    else:
        if -diff < margin:
            delta = 0.5 * (margin + diff)
            block[3, 4] = c1 - delta
            block[4, 3] = c2 + delta
    return block

def embed_watermark_dwt_dct(Y_channel, ref_map, alpha=ALPHA):
    """
    Pairwise margin embedding with redundancy.
    ref_map in {-1,+1} shape (s,s)
    """
    coeffs = pywt.wavedec2(Y_channel, MODEL, level=LEVEL)
    LL = coeffs[0]
    dct_ll = apply_dct(LL)

    base_strength = _coeff_strength_from_ll(dct_ll)
    # NEW: enforce a minimum margin so flat images still encode reliably
    MIN_MARGIN = 5.0
    margin = max(alpha * base_strength, MIN_MARGIN)

    bits = ref_map.flatten()  # {-1,+1}
    h, w = dct_ll.shape
    total_blocks = (h // 8) * (w // 8)
    needed_blocks = int(bits.size * REDUNDANCY)

    idx = 0  # block counter
    bit_idx = 0
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            if bit_idx >= bits.size or idx >= needed_blocks:
                break
            blk = dct_ll[i:i+8, j:j+8]
            blk = _embed_pair_margin(blk, bits[bit_idx], margin)
            dct_ll[i:i+8, j:j+8] = blk
            idx += 1
            if idx % REDUNDANCY == 0:
                bit_idx += 1
        if bit_idx >= bits.size or idx >= needed_blocks:
            break

    coeffs[0] = inverse_dct(dct_ll)
    Y_watermarked = pywt.waverec2(coeffs, MODEL)
    return Y_watermarked[:Y_channel.shape[0], :Y_channel.shape[1]]

def extract_watermark_dwt_dct(Y_channel, wm_size):
    """
    Majority-vote extraction on pairwise differences.
    Returns map in {-1,+1} of shape (wm_size, wm_size)
    """
    coeffs = pywt.wavedec2(Y_channel, MODEL, level=LEVEL)
    LL = coeffs[0]
    dct_ll = apply_dct(LL)

    h, w = dct_ll.shape
    votes = np.zeros((wm_size * wm_size,), dtype=np.int32)

    idx = 0
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            bit_index = idx // REDUNDANCY
            if bit_index >= votes.size:
                break
            blk = dct_ll[i:i+8, j:j+8]
            diff = blk[3, 4] - blk[4, 3]
            votes[bit_index] += 1 if diff >= 0 else -1
            idx += 1
        if (idx // REDUNDANCY) >= votes.size:
            break

    bits = np.where(votes >= 0, 1.0, -1.0)
    return bits.reshape(wm_size, wm_size)

def embed_watermark_color(host_rgb, watermark_gray, alpha=ALPHA):
    """
    Capacity-aware embedding pipeline. Returns (watermarked_rgb, ref_map{-1,+1}, wm_size).
    """
    Y, Cr, Cb = rgb_to_ycbcr(host_rgb)
    ref_map, wm_size, _ = _prepare_capacity_and_wm(Y, watermark_gray)
    Y_wm = embed_watermark_dwt_dct(Y, ref_map, alpha)
    return ycbcr_to_rgb(Y_wm, Cr, Cb), ref_map, wm_size

def extract_watermark_color(wm_rgb, wm_size):
    """
    Extraction pipeline using known wm_size and redundancy.
    """
    Y, _, _ = rgb_to_ycbcr(wm_rgb)
    return extract_watermark_dwt_dct(Y, wm_size)

# =========================================================================
# === ATTACK FUNCTIONS
# =========================================================================

def attack_jpeg(img_rgb, quality):
    img_pil = Image.fromarray(img_rgb)
    path = f'./attacks/jpeg_q{quality}.jpg'
    img_pil.save(path, 'JPEG', quality=quality)
    return np.array(Image.open(path))

def attack_noise(img_rgb, sigma):
    noise = np.random.normal(0, sigma, img_rgb.shape)
    attacked = np.clip(img_rgb + noise * 255, 0, 255).astype(np.uint8)
    Image.fromarray(attacked).save(f'./attacks/noise_s{sigma:.2f}.jpg')
    return attacked

def attack_blur(img_rgb, sigma):
    attacked = np.zeros_like(img_rgb)
    for c in range(3):
        attacked[:,:,c] = gaussian_filter(img_rgb[:,:,c], sigma=sigma)
    attacked = np.clip(attacked, 0, 255).astype(np.uint8)
    Image.fromarray(attacked).save(f'./attacks/blur_s{sigma:.1f}.jpg')
    return attacked

# =========================================================================
# === METRICS
# =========================================================================

def measure_psnr(original, watermarked):
    mse = np.mean((original.astype(np.float64) - watermarked.astype(np.float64)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def measure_ncc(original_wm, recovered_wm):
    """
    Robust correlation on {-1,+1} maps.
    Falls back to sign product mean to avoid NaN on zero-variance arrays.
    """
    a = np.sign(np.asarray(original_wm, dtype=np.float64).flatten())
    b = np.sign(np.asarray(recovered_wm, dtype=np.float64).flatten())
    # Ensure {-1,+1}
    a[a == 0] = 1.0
    b[b == 0] = 1.0
    # Correlation-like score in [-1, 1]
    return float(np.mean(a * b))

# =========================================================================
# === MAIN EVALUATION
# =========================================================================

def run_evaluation():
    print("="*60)
    print(" DWT+DCT WATERMARKING EVALUATION")
    print(f" Mode: {WATERMARK_MODE.upper()}")
    print("="*60)
    
    host_rgb = load_image(IMAGE_HOST, HOST_SIZE)
    print(f"\n✓ Host image loaded: {host_rgb.shape}")
    
    # Prepare watermark (raw grayscale 0..255)
    if WATERMARK_MODE == 'image':
        if not os.path.exists(IMAGE_WATERMARK):
            print(f"❌ ERROR: Watermark image not found: {IMAGE_WATERMARK}")
            return
        wm_gray_raw = load_watermark_image(IMAGE_WATERMARK, 256)  # initial size, will be resized to capacity
        print(f"✓ Image watermark loaded (pre): {wm_gray_raw.shape}")
    elif WATERMARK_MODE == 'text':
        wm_gray_raw = create_text_watermark(WATERMARK_TEXT, 256)
        print(f"✓ Text watermark created: '{WATERMARK_TEXT}'")
        Image.fromarray(np.uint8(wm_gray_raw)).save('./result/text_watermark.png')
    else:
        print(f"❌ ERROR: Invalid mode '{WATERMARK_MODE}'. Use 'image' or 'text'.")
        return
    
    # Embed (capacity-aware)
    wm_host_rgb, ref_map, wm_size = embed_watermark_color(host_rgb, wm_gray_raw, ALPHA)
    Image.fromarray(wm_host_rgb).save('./result/watermarked.png')
    print(f"✓ Watermarked image saved: ./result/watermarked.png")
    print(f"Effective watermark size (capacity-matched, redundancy={REDUNDANCY}): {wm_size}x{wm_size}")
    
    # --- IMPERCEPTIBILITY ---
    print("\n" + "="*60)
    print(" IMPERCEPTIBILITY TEST")
    print("="*60)
    psnr = measure_psnr(host_rgb, wm_host_rgb)
    print(f"PSNR: {psnr:.2f} dB")
    if psnr >= 40:
        print("✅ EXCELLENT - Watermark is invisible")
    elif psnr >= 35:
        print("✅ GOOD - Watermark is barely noticeable")
    else:
        print("⚠️  FAIR - Some degradation visible")
    
    # --- ROBUSTNESS ---
    print("\n" + "="*60)
    print(" ROBUSTNESS TEST (NCC)")
    print("="*60)
    
    # Baseline
    extracted_baseline = extract_watermark_color(wm_host_rgb, wm_size)  # {-1,+1}
    vis = ((extracted_baseline + 1.0) * 0.5 * 255.0).astype(np.uint8)
    Image.fromarray(vis).save('./result/extracted_baseline.png')
    ncc_baseline = measure_ncc(ref_map, extracted_baseline)
    print(f"Baseline (No Attack):           NCC = {ncc_baseline:.4f}")
    
    # Attacks
    attacks = [
        ('JPEG Q=85', lambda: attack_jpeg(wm_host_rgb, 85)),
        ('JPEG Q=70', lambda: attack_jpeg(wm_host_rgb, 70)),
        ('JPEG Q=50', lambda: attack_jpeg(wm_host_rgb, 50)),
        ('Noise σ=0.03', lambda: attack_noise(wm_host_rgb, 0.03)),
        ('Noise σ=0.06', lambda: attack_noise(wm_host_rgb, 0.06)),
        ('Blur σ=0.8', lambda: attack_blur(wm_host_rgb, 0.8)),
        ('Blur σ=1.2', lambda: attack_blur(wm_host_rgb, 1.2)),
    ]
    
    ncc_scores = []
    for name, attack_fn in attacks:
        attacked = attack_fn()
        extracted = extract_watermark_color(attacked, wm_size)
        ncc = measure_ncc(ref_map, extracted)
        ncc_scores.append(ncc)
        print(f"{name:<25} NCC = {ncc:.4f}")
    
    avg_ncc = float(np.mean(ncc_scores))
    print(f"\n{'Average NCC (Attacks):':<25} {avg_ncc:.4f}")
    
    if avg_ncc >= 0.7:
        print("✅ HIGHLY ROBUST - Survives most attacks")
    elif avg_ncc >= 0.5:
        print("✅ MODERATELY ROBUST - Survives light attacks")
    else:
        print("⚠️  WEAK - Watermark easily destroyed")
    
    print("\n" + "="*60)
    print(" EVALUATION COMPLETE")
    print("="*60)
    print(f"Mode: {WATERMARK_MODE}")
    print(f"Watermarked: ./result/watermarked.png")
    print(f"Extracted: ./result/extracted_baseline.png")

if __name__ == "__main__":
    run_evaluation()