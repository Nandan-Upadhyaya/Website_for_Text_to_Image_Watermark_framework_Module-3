"""
Invisible Watermarking Module for AI-Image-Suite
DWT-DCT based invisible watermarking with robust extraction
"""
import numpy as np
import pywt
from PIL import Image, ImageDraw, ImageFont
from scipy.fftpack import dct, idct
from scipy.ndimage import gaussian_filter
import cv2
import io
import base64

# Configuration (LOCKED)
MODEL = 'haar'
LEVEL = 1
ALPHA = 0.38
REDUNDANCY = 6

# =========================
# Helpers
# =========================
def rgb_to_ycbcr(img_rgb):
    ycbcr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    return ycbcr[:, :, 0].astype(np.float64), ycbcr[:, :, 1], ycbcr[:, :, 2]

def ycbcr_to_rgb(Y, Cr, Cb):
    Y_uint = np.clip(Y, 0, 255).astype(np.uint8)
    ycbcr = np.stack([Y_uint, Cr, Cb], axis=2)
    return cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2RGB)

def create_text_watermark(text, size, font_size=None):
    if font_size is None:
        font_size = max(12, size // 8)
    img = Image.new('L', (size, size), color=0)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except Exception:
        text_w, text_h = draw.textsize(text, font=font)
    x = (size - text_w) // 2
    y = (size - text_h) // 2
    draw.text((x, y), text, fill=255, font=font)
    return np.array(img, dtype=np.float64)

def load_watermark_image_from_pil(pil_image, size):
    img = pil_image.convert('L')
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return np.array(img, dtype=np.float64)

def apply_dct(image_array):
    h, w = image_array.shape
    h = (h // 8) * 8
    w = (w // 8) * 8
    image_array = image_array[:h, :w]
    out = np.zeros_like(image_array)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            blk = image_array[i:i+8, j:j+8]
            out[i:i+8, j:j+8] = dct(dct(blk.T, norm="ortho").T, norm="ortho")
    return out

def inverse_dct(dct_array):
    h, w = dct_array.shape
    out = np.zeros_like(dct_array)
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            blk = dct_array[i:i+8, j:j+8]
            out[i:i+8, j:j+8] = idct(idct(blk.T, norm="ortho").T, norm="ortho")
    return out

def _prepare_capacity_and_wm(Y_channel, wm_gray):
    coeffs = pywt.wavedec2(Y_channel, MODEL, level=LEVEL)
    LL = coeffs[0]
    h, w = LL.shape
    blocks_h, blocks_w = (h // 8), (w // 8)
    cap_blocks = blocks_h * blocks_w

    redundancy = REDUNDANCY
    s = int(np.floor(np.sqrt(max(1, cap_blocks // redundancy))))
    if s < 4:
        s = 4
        redundancy = max(1, cap_blocks // (s * s))

    wm_img = Image.fromarray(np.uint8(np.clip(wm_gray, 0, 255)))
    wm_resized = wm_img.resize((s, s), Image.Resampling.LANCZOS)
    wm_arr = np.array(wm_resized, dtype=np.float64)
    thr = float(np.median(wm_arr))
    bits01 = (wm_arr >= thr).astype(np.uint8)
    ref_map = (bits01 * 2 - 1).astype(np.float64)
    return ref_map, s, (h, w), redundancy

def _coeff_strength_from_ll(dct_ll):
    a = np.abs(dct_ll[3::8, 4::8]).flatten()
    b = np.abs(dct_ll[4::8, 3::8]).flatten()
    return np.percentile(np.concatenate([a, b]), 75) + 1e-6

def _embed_pair_margin(block, bit, margin):
    c1 = block[3, 4]
    c2 = block[4, 3]
    diff = c1 - c2
    if bit > 0:
        if diff < margin:
            delta = 0.52 * (margin - diff)
            block[3, 4] = c1 + delta
            block[4, 3] = c2 - delta
    else:
        if -diff < margin:
            delta = 0.52 * (margin + diff)
            block[3, 4] = c1 - delta
            block[4, 3] = c2 + delta
    return block

def embed_watermark_dwt_dct(Y_channel, ref_map, alpha=ALPHA, redundancy=REDUNDANCY):
    coeffs = pywt.wavedec2(Y_channel, MODEL, level=LEVEL)
    LL = coeffs[0]
    dct_ll = apply_dct(LL)

    base_strength = _coeff_strength_from_ll(dct_ll)
    MIN_MARGIN = 6.0
    margin = max(alpha * base_strength, MIN_MARGIN)
    if base_strength > 18.0:
        margin *= 1.05

    bits = ref_map.flatten()
    h, w = dct_ll.shape
    needed_blocks = int(bits.size * redundancy)
    idx = 0
    bit_idx = 0
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            if bit_idx >= bits.size or idx >= needed_blocks:
                break
            blk = dct_ll[i:i+8, j:j+8]
            dct_ll[i:i+8, j:j+8] = _embed_pair_margin(blk, bits[bit_idx], margin)
            idx += 1
            if idx % redundancy == 0:
                bit_idx += 1
        if bit_idx >= bits.size or idx >= needed_blocks:
            break

    coeffs[0] = inverse_dct(dct_ll)
    Y_wm = pywt.waverec2(coeffs, MODEL)
    return Y_wm[:Y_channel.shape[0], :Y_channel.shape[1]]

def extract_watermark_dwt_dct(Y_channel, wm_size, redundancy=REDUNDANCY):
    coeffs = pywt.wavedec2(Y_channel, MODEL, level=LEVEL)
    LL = coeffs[0]
    dct_ll = apply_dct(LL)

    h, w = dct_ll.shape
    votes = np.zeros((wm_size * wm_size,), dtype=np.int32)
    idx = 0
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            bit_index = idx // redundancy
            if bit_index >= votes.size:
                break
            blk = dct_ll[i:i+8, j:j+8]
            diff = blk[3, 4] - blk[4, 3]
            votes[bit_index] += 1 if diff >= 0 else -1
            idx += 1
        if (idx // redundancy) >= votes.size:
            break

    bits = np.where(votes >= 0, 1.0, -1.0)
    return bits.reshape(wm_size, wm_size)

def embed_watermark_color(host_rgb, watermark_gray, alpha=ALPHA):
    Y, Cr, Cb = rgb_to_ycbcr(host_rgb)
    ref_map, wm_size, _, redundancy = _prepare_capacity_and_wm(Y, watermark_gray)
    Y_wm = embed_watermark_dwt_dct(Y, ref_map, alpha, redundancy)
    return ycbcr_to_rgb(Y_wm, Cr, Cb), ref_map, wm_size, redundancy

def extract_watermark_color(wm_rgb, wm_size, redundancy=REDUNDANCY):
    Y, _, _ = rgb_to_ycbcr(wm_rgb)
    return extract_watermark_dwt_dct(Y, wm_size, redundancy)

# =========================
# Attacks and metrics
# =========================
def attack_jpeg(img_rgb, quality):
    img_pil = Image.fromarray(img_rgb)
    buffer = io.BytesIO()
    img_pil.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    return np.array(Image.open(buffer))

def attack_noise(img_rgb, sigma):
    noise = np.random.normal(0, sigma, img_rgb.shape)
    attacked = np.clip(img_rgb + noise * 255, 0, 255).astype(np.uint8)
    return attacked

def attack_blur(img_rgb, sigma):
    attacked = np.zeros_like(img_rgb, dtype=np.float64)
    for c in range(3):
        attacked[:, :, c] = gaussian_filter(img_rgb[:, :, c].astype(np.float64), sigma=sigma)
    attacked = np.clip(attacked, 0, 255).astype(np.uint8)
    return attacked

def measure_psnr(original, watermarked):
    mse = np.mean((original.astype(np.float64) - watermarked.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))

def measure_ncc(original_wm, recovered_wm):
    a = np.sign(np.asarray(original_wm, dtype=np.float64).flatten())
    b = np.sign(np.asarray(recovered_wm, dtype=np.float64).flatten())
    a[a == 0] = 1.0
    b[b == 0] = 1.0
    return float(np.mean(a * b))

# =========================
# Public API
# =========================
def apply_invisible_watermark(host_pil_image, watermark_mode, watermark_text=None,
                              watermark_pil_image=None, alpha=None):
    """
    Returns a PIL Image only (no tuple) to remain compatible with callers that call .save().
    """
    alpha = ALPHA
    host_rgb = np.array(host_pil_image.convert('RGB'), dtype=np.uint8)
    if watermark_mode == 'text':
        wm_gray = create_text_watermark(watermark_text or 'Copyright', 256)
    else:
        wm_gray = load_watermark_image_from_pil(watermark_pil_image, 256)
    wm_host_rgb, _, _, _ = embed_watermark_color(host_rgb, wm_gray, alpha)
    result = Image.fromarray(wm_host_rgb)
    # Optionally attach meta (non-breaking)
    try:
        psnr = measure_psnr(host_rgb, wm_host_rgb)
        result.info['imperceptibility_psnr'] = round(float(psnr), 2)
    except Exception:
        pass
    return result

def test_watermark_robustness(watermarked_pil_image, watermark_mode, watermark_text=None,
                              watermark_pil_image=None, alpha=None):
    """
    Compute imperceptibility PSNR exactly as in watermarkdwt.py:
    PSNR(original_host_rgb, watermarked_rgb), then run attacks and return results.
    """
    alpha = ALPHA
    # Treat provided image as ORIGINAL HOST
    if watermarked_pil_image.mode != 'RGB':
        watermarked_pil_image = watermarked_pil_image.convert('RGB')
    host_rgb = np.array(watermarked_pil_image, dtype=np.uint8)

    # Build watermark content
    if watermark_mode == 'text':
        wm_gray_raw = create_text_watermark(watermark_text or 'Copyright', 256)
    else:
        if watermark_pil_image.mode != 'L':
            watermark_pil_image = watermark_pil_image.convert('L')
        wm_gray_raw = np.array(watermark_pil_image, dtype=np.float64)

    # Capacity and ref map (based on HOST)
    Y_host = cv2.cvtColor(host_rgb, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float64)
    ref_map, wm_size, _, redundancy = _prepare_capacity_and_wm(Y_host, wm_gray_raw)

    # Embed first to obtain watermarked image (for PSNR and attacks)
    wm_host_rgb, _, _, _ = embed_watermark_color(host_rgb, wm_gray_raw, alpha)
    imperceptibility_psnr = measure_psnr(host_rgb, wm_host_rgb)
    print(f"   ✅ Imperceptibility PSNR: {imperceptibility_psnr:.2f} dB (computed from original vs watermarked)")

    wm_binary = ref_map
    results = []

    # Test 1: Original (no attack)
    try:
        recovered_wm_orig = extract_watermark_color(wm_host_rgb, wm_size, redundancy)
        ncc_orig = measure_ncc(wm_binary, recovered_wm_orig)
        results.append({
            'attack': 'Original (No Attack)',
            'psnr': 0.0,
            'ncc': round(ncc_orig, 4),
            'success': ncc_orig > 0.6
        })
        print(f"   ✓ Original: NCC = {ncc_orig:.4f}")
    except Exception as e:
        results.append({
            'attack': 'Original (No Attack)',
            'psnr': 0.0,
            'ncc': 0.0,
            'success': False,
            'error': str(e)
        })

    # JPEG
    for quality in [85, 70, 50]:
        try:
            attacked_rgb = attack_jpeg(wm_host_rgb, quality)
            psnr = measure_psnr(wm_host_rgb, attacked_rgb)
            recovered = extract_watermark_color(attacked_rgb, wm_size, redundancy)
            ncc = measure_ncc(wm_binary, recovered)
            results.append({
                'attack': f'JPEG Compression (Q={quality})',
                'psnr': round(psnr, 2),
                'ncc': round(ncc, 4),
                'success': ncc > 0.6
            })
            print(f"   ✓ JPEG Q={quality}: PSNR = {psnr:.2f} dB, NCC = {ncc:.4f}")
        except Exception as e:
            results.append({
                'attack': f'JPEG Compression (Q={quality})',
                'psnr': 0.0,
                'ncc': 0.0,
                'success': False,
                'error': str(e)
            })

    # Noise
    for sigma in [0.03, 0.06]:
        try:
            attacked_rgb = attack_noise(wm_host_rgb, sigma)
            psnr = measure_psnr(wm_host_rgb, attacked_rgb)
            recovered = extract_watermark_color(attacked_rgb, wm_size, redundancy)
            ncc = measure_ncc(wm_binary, recovered)
            results.append({
                'attack': f'Gaussian Noise (σ={sigma})',
                'psnr': round(psnr, 2),
                'ncc': round(ncc, 4),
                'success': ncc > 0.6
            })
            print(f"   ✓ Noise σ={sigma}: PSNR = {psnr:.2f} dB, NCC = {ncc:.4f}")
        except Exception as e:
            results.append({
                'attack': f'Gaussian Noise (σ={sigma})',
                'psnr': 0.0,
                'ncc': 0.0,
                'success': False,
                'error': str(e)
            })

    # Blur
    for sigma in [0.8, 1.2]:
        try:
            attacked_rgb = attack_blur(wm_host_rgb, sigma)
            psnr = measure_psnr(wm_host_rgb, attacked_rgb)
            recovered = extract_watermark_color(attacked_rgb, wm_size, redundancy)
            ncc = measure_ncc(wm_binary, recovered)
            results.append({
                'attack': f'Gaussian Blur (σ={sigma})',
                'psnr': round(psnr, 2),
                'ncc': round(ncc, 4),
                'success': ncc > 0.6
            })
            print(f"   ✓ Blur σ={sigma}: PSNR = {psnr:.2f} dB, NCC = {ncc:.4f}")
        except Exception as e:
            results.append({
                'attack': f'Gaussian Blur (σ={sigma})',
                'psnr': 0.0,
                'ncc': 0.0,
                'success': False,
                'error': str(e)
            })

    # Encode watermarked image for UI
    buf = io.BytesIO()
    Image.fromarray(wm_host_rgb).save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return {
        'results': results,
        'original_image': f'data:image/png;base64,{img_b64}',
        'watermark_size': int(wm_size),
        'redundancy': int(redundancy),
        'alpha': float(alpha),
        'imperceptibility_psnr': round(float(imperceptibility_psnr), 2)
    }

# =========================
# MAIN EVALUATION (for CLI)
# =========================
def run_evaluation():
    import os
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
        wm_gray_raw = load_watermark_image(IMAGE_WATERMARK, 256)
        print(f"✓ Image watermark loaded (pre): {wm_gray_raw.shape}")
    elif WATERMARK_MODE == 'text':
        wm_gray_raw = create_text_watermark(WATERMARK_TEXT, 256)
        print(f"✓ Text watermark created: '{WATERMARK_TEXT}'")
        Image.fromarray(np.uint8(wm_gray_raw)).save('./result/text_watermark.png')
    else:
        print(f"❌ ERROR: Invalid mode '{WATERMARK_MODE}'. Use 'image' or 'text'.")
        return
    
    # Embed (capacity-aware)
    wm_host_rgb, ref_map, wm_size, redundancy = embed_watermark_color(host_rgb, wm_gray_raw, ALPHA)
    Image.fromarray(wm_host_rgb).save('./result/watermarked.png')
    print(f"✓ Watermarked image saved: ./result/watermarked.png")
    print(f"Effective watermark size (capacity-matched, redundancy={redundancy}): {wm_size}x{wm_size}")
    
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
    extracted_baseline = extract_watermark_color(wm_host_rgb, wm_size, redundancy)
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
        extracted = extract_watermark_color(attacked, wm_size, redundancy)
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