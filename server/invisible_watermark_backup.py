"""
Invisible Watermarking Module for AI-Image-Suite
DWT-DCT based invisible watermarking
"""
import numpy as np
import pywt
from PIL import Image, ImageDraw, ImageFont
from scipy.fftpack import dct, idct
from scipy.ndimage import gaussian_filter
import cv2
import io
import base64

# Configuration
MODEL = 'haar'
LEVEL = 1
ALPHA = 0.28
REDUNDANCY = 8

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
    Adaptive: reduces redundancy for smaller images.
    Returns: ref_map {-1,+1} of shape (s,s), s, (h,w), actual_redundancy
    """
    coeffs = pywt.wavedec2(Y_channel, MODEL, level=LEVEL)
    LL = coeffs[0]
    h, w = LL.shape
    blocks_h, blocks_w = (h // 8), (w // 8)
    cap_blocks = blocks_h * blocks_w
    
    # Adaptively determine redundancy and watermark size
    redundancy = REDUNDANCY
    s = int(np.floor(np.sqrt(max(1, cap_blocks // redundancy))))
    
    # If image is too small, reduce redundancy
    while s < 8 and redundancy > 1:
        redundancy = max(1, redundancy // 2)
        s = int(np.floor(np.sqrt(max(1, cap_blocks // redundancy))))
    
    # Minimum watermark size
    if s < 4:
        s = 4  # Absolute minimum
        redundancy = max(1, cap_blocks // (s * s))
    
    # Resize watermark to s x s
    wm_img = Image.fromarray(np.uint8(np.clip(wm_gray, 0, 255)))
    wm_resized = wm_img.resize((s, s), Image.Resampling.LANCZOS)
    wm_arr = np.array(wm_resized, dtype=np.float64)
    # Binarize using adaptive threshold (median is robust)
    thr = float(np.median(wm_arr))
    bits01 = (wm_arr >= thr).astype(np.uint8)
    ref_map = (bits01 * 2 - 1).astype(np.float64)  # {-1, +1}
    return ref_map, s, (h, w), redundancy

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

def embed_watermark_dwt_dct(Y_channel, ref_map, alpha=ALPHA, redundancy=REDUNDANCY):
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
    needed_blocks = int(bits.size * redundancy)

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
            if idx % redundancy == 0:
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
    ref_map, wm_size, _, redundancy = _prepare_capacity_and_wm(Y, watermark_gray)
    Y_wm = embed_watermark_dwt_dct(Y, ref_map, alpha, redundancy)
    return ycbcr_to_rgb(Y_wm, Cr, Cb), ref_map, wm_size

def extract_watermark_color(wm_rgb, wm_size):
    """
    Extraction pipeline using known wm_size and redundancy.
    """
    Y, _, _ = rgb_to_ycbcr(wm_rgb)
    return extract_watermark_dwt_dct(Y, wm_size)

# =========================================================================
# === PUBLIC API FOR INTEGRATION
# =========================================================================

def apply_invisible_watermark(host_pil_image, watermark_mode='text', watermark_text=None, watermark_pil_image=None, alpha=ALPHA):
    """
    Apply invisible watermark to a PIL Image.
    
    Args:
        host_pil_image: PIL Image to watermark
        watermark_mode: 'text' or 'image'
        watermark_text: Text to embed (if mode='text')
        watermark_pil_image: PIL Image watermark (if mode='image')
        alpha: Embedding strength (default 0.28)
    
    Returns:
        PIL Image with invisible watermark embedded
    """
    # Convert host to RGB numpy array
    host_rgb = np.array(host_pil_image.convert('RGB'), dtype=np.uint8)
    
    # Prepare watermark
    if watermark_mode == 'text':
        if not watermark_text:
            raise ValueError("watermark_text required for text mode")
        wm_gray = create_text_watermark(watermark_text, 256)
    elif watermark_mode == 'image':
        if watermark_pil_image is None:
            raise ValueError("watermark_pil_image required for image mode")
        wm_gray = load_watermark_image_from_pil(watermark_pil_image, 256)
    else:
        raise ValueError(f"Invalid watermark_mode: {watermark_mode}")
    
    # Embed watermark
    wm_host_rgb, ref_map, wm_size = embed_watermark_color(host_rgb, wm_gray, alpha)
    
    # Convert back to PIL Image
    result = Image.fromarray(wm_host_rgb)
    
    return result

def load_watermark_image_from_pil(pil_image, size):
    """Load watermark from PIL Image as grayscale"""
    img = pil_image.convert('L')
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return np.array(img, dtype=np.float64)

# =========================================================================
# === ATTACK FUNCTIONS (NOT USED IN INTEGRATION)
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


def test_watermark_robustness(watermarked_pil_image, watermark_mode, watermark_text=None, 
                               watermark_pil_image=None, alpha=0.28):
    """
    Test the robustness of an invisible watermark against various attacks.
    
    Args:
        watermarked_pil_image: PIL Image with embedded watermark
        watermark_mode: 'text' or 'image'
        watermark_text: Text watermark (if mode is 'text')
        watermark_pil_image: PIL Image watermark (if mode is 'image')
        alpha: Embedding strength used (default 0.28)
    
    Returns:
        dict: Results containing attack names and metrics
              {
                  'results': [
                      {'attack': 'Original', 'psnr': 45.2, 'ncc': 1.0, 'success': True},
                      {'attack': 'JPEG Q=85', 'psnr': 42.5, 'ncc': 0.95, 'success': True},
                      ...
                  ],
                  'original_image': base64 encoded original watermarked image
              }
    """
    print(f"Starting robustness testing...")
    print(f"Watermark mode: {watermark_mode}")
    
    # Convert watermarked image to RGB
    if watermarked_pil_image.mode != 'RGB':
        watermarked_pil_image = watermarked_pil_image.convert('RGB')
    
    wm_rgb = np.array(watermarked_pil_image, dtype=np.float32)
    
    # Create the original watermark first (before preparing capacity)
    if watermark_mode == 'text':
        wm_gray_raw = create_text_watermark(watermark_text, 256)
    else:  # image
        if watermark_pil_image.mode != 'L':
            watermark_pil_image = watermark_pil_image.convert('L')
        wm_gray_raw = np.array(watermark_pil_image, dtype=np.float32)
    
    # Prepare reference watermark and map
    Y = cv2.cvtColor(wm_rgb.astype(np.uint8), cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)
    ref_map, wm_size, _, redundancy = _prepare_capacity_and_wm(Y, wm_gray_raw)
    
    # The ref_map is already the resized and binarized watermark from _prepare_capacity_and_wm
    wm_binary = ref_map  # ref_map is already in {-1, +1} format
    
    results = []
    
    # Test 1: Original (no attack)
    try:
        recovered_wm_orig = extract_watermark_color(wm_rgb, wm_size)
        ncc_orig = measure_ncc(wm_binary, recovered_wm_orig)
        results.append({
            'attack': 'Original (No Attack)',
            'psnr': 0.0,  # Not applicable for original
            'ncc': round(ncc_orig, 4),
            'success': ncc_orig > 0.6
        })
        print(f"✓ Original: NCC = {ncc_orig:.4f}")
    except Exception as e:
        print(f"✗ Original test failed: {e}")
        results.append({
            'attack': 'Original (No Attack)',
            'psnr': 0.0,
            'ncc': 0.0,
            'success': False,
            'error': str(e)
        })
    
    # Test 2: JPEG Compression (various qualities)
    for quality in [85, 70, 50]:
        try:
            attacked_rgb = attack_jpeg(wm_rgb, quality)
            psnr = measure_psnr(wm_rgb, attacked_rgb)
            recovered_wm = extract_watermark_color(attacked_rgb, wm_size)
            ncc = measure_ncc(wm_binary, recovered_wm)
            results.append({
                'attack': f'JPEG Compression (Q={quality})',
                'psnr': round(psnr, 2),
                'ncc': round(ncc, 4),
                'success': ncc > 0.6
            })
            print(f"✓ JPEG Q={quality}: PSNR = {psnr:.2f} dB, NCC = {ncc:.4f}")
        except Exception as e:
            print(f"✗ JPEG Q={quality} test failed: {e}")
            results.append({
                'attack': f'JPEG Compression (Q={quality})',
                'psnr': 0.0,
                'ncc': 0.0,
                'success': False,
                'error': str(e)
            })
    
    # Test 3: Gaussian Noise
    for sigma in [0.03, 0.06]:
        try:
            attacked_rgb = attack_noise(wm_rgb, sigma)
            psnr = measure_psnr(wm_rgb, attacked_rgb)
            recovered_wm = extract_watermark_color(attacked_rgb, wm_size)
            ncc = measure_ncc(wm_binary, recovered_wm)
            results.append({
                'attack': f'Gaussian Noise (σ={sigma})',
                'psnr': round(psnr, 2),
                'ncc': round(ncc, 4),
                'success': ncc > 0.6
            })
            print(f"✓ Noise σ={sigma}: PSNR = {psnr:.2f} dB, NCC = {ncc:.4f}")
        except Exception as e:
            print(f"✗ Noise σ={sigma} test failed: {e}")
            results.append({
                'attack': f'Gaussian Noise (σ={sigma})',
                'psnr': 0.0,
                'ncc': 0.0,
                'success': False,
                'error': str(e)
            })
    
    # Test 4: Gaussian Blur
    for sigma in [0.8, 1.2]:
        try:
            attacked_rgb = attack_blur(wm_rgb, sigma)
            psnr = measure_psnr(wm_rgb, attacked_rgb)
            recovered_wm = extract_watermark_color(attacked_rgb, wm_size)
            ncc = measure_ncc(wm_binary, recovered_wm)
            results.append({
                'attack': f'Gaussian Blur (σ={sigma})',
                'psnr': round(psnr, 2),
                'ncc': round(ncc, 4),
                'success': ncc > 0.6
            })
            print(f"✓ Blur σ={sigma}: PSNR = {psnr:.2f} dB, NCC = {ncc:.4f}")
        except Exception as e:
            print(f"✗ Blur σ={sigma} test failed: {e}")
            results.append({
                'attack': f'Gaussian Blur (σ={sigma})',
                'psnr': 0.0,
                'ncc': 0.0,
                'success': False,
                'error': str(e)
            })
    
    # Convert original watermarked image to base64 for frontend
    buffered = io.BytesIO()
    watermarked_pil_image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    print(f"Robustness testing completed. {len(results)} tests performed.")
    
    return {
        'results': results,
        'original_image': f'data:image/png;base64,{img_base64}',
        'watermark_size': wm_size,
        'redundancy': redundancy,
        'alpha': alpha
    }


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