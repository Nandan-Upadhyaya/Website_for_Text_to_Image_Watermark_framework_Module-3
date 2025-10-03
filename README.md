# AI Image Processing Suite

A comprehensive Text Prompt to Watermarked Image generation web application that integrates three powerful modules:

1. **DF-GAN Text-to-Image Generation (Module 1)** - Generate stunning images from text descriptions
2. **Image Prompt Evaluator (Module 2)** - Evaluate how well AI-generated images match their text prompts
3. **Watermark Protection UI (Module 3)** - Add professional watermarks to protect your images
4. **Website Integration (Full Suite)** - A cohesive web app that orchestrates DF-GAN generation (Module 1), CLIP-based evaluation (Module 2), and watermarking (Module 3), adding batch workflows, auto-evaluation, galleries, recommendations, and streamlined downloads.

## Features

### 🎨 Text-to-Image Generation
- Generate images from text descriptions using the DF-GAN model
- Support for multiple datasets (CUB, COCO)
- Customizable generation settings (batch size, steps, guidance scale, seed)
- Example prompts for different datasets
- Real-time generation progress tracking

### 🔍 Image Quality Evaluation
- CLIP-based semantic similarity evaluation
- Keyword analysis and confidence scoring
- Adjustable similarity thresholds
- Quality classification (Excellent, Good, Fair, Poor)
- Detailed feedback and suggestions

### 🛡️ Watermark Protection
- Bulk image watermarking
- Customizable watermark positioning and opacity
- Auto-resizing and padding options
- File naming with prefix/suffix support
- Batch download functionality

### 📱 Modern UI/UX
- Responsive design for all device sizes
- Drag-and-drop file uploads
- Real-time progress indicators
- Real time Toast notification integration for flow indication
- Image gallery with filtering and search

## Technology Stack

- **Frontend**: React 18 with functional components and hooks
- **Styling**: Tailwind CSS with custom components
- **Icons**: Heroicons
- **Routing**: React Router DOM
- **File Handling**: React Dropzone
- **Notifications**: React Hot Toast
- **Build Tool**: Create React App

## Server Setup and Path Configuration

The AI-Image-Suite server uses a flexible path configuration system that works across different development environments and systems.

### Quick Setup

1. **Clone the repositories**
   ```bash
   # Clone the main project
   git clone AI-Image-Suite
   cd AI-Image-Suite
   

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install dependencies**
   ```bash
   npm install
   ```


### Validation

Check if your setup is correct:

# Start the server
python server/app.py

# Check the setup endpoint
curl http://localhost:5001/api/check


The `/api/check` endpoint will report any missing files or incorrect paths.

# In one more terminal:

3. cd AI-Image-Suite

4. **Start the development server**
   ```bash
   npm start
   ```

5. **Open your browser**
   Navigate to `http://localhost:3000`


## Modules at a Glance (Standalone References)
- Module 1: DF-GAN Text-to-Image Generation
  - GitHub: https://github.com/The-GANners/DF-GAN
  - Purpose: Generate images from text using GANs (CUB/COCO, etc.)
- Module 2: CLIP-based Image Prompt Evaluator
  - Github: https://github.com/The-GANners/Module-2
  - Purpose: Score semantic match between an image and prompt + keyword/feature analysis
- Module 3: Watermark UI (reference implementation)
  - Github: https://github.com/The-GANners/Watermark-UI
  - Purpose: Add batch watermarks with placement/opacity/scale controls

## How Each Module Works

### Module 1 — DF-GAN Text-to-Image (Standalone)
- What it is:
  - A text-to-image GAN that synthesizes images from natural language prompts. Training/evaluation entry points are under code/src (train.py, sample.py, test_FID.py) with YAML configs in code/cfg.

- Inputs:
  - Prompt(s): tokenized via DAMSM vocabulary.
  - Noise Vector: z_dim (default 100), batch size, truncation/trunc_rate, manual_seed, encoder_epoch; dataset-specific text/image encoders.

- Outputs:
  - Generated PNGs (normalized from [-1,1] to [0,255]) saved per-sentence. Training also writes periodic grids and checkpoints to saved_models folder.
  - Metrics/artifacts: FID computed with 2048-dim Inception features against npz stats.
  - optional CLIP alignment grid and score saved under alignment_samples.

- Key components:
  - Text Encoder : bi-LSTM that returns word- and sentence-level embeddings (256).

  - Image encoder : (CNN_ENCODER): Inception v3 backbone projecting image features to 256-d, used by DAMSM and dataset prep.

  - Generator:
    - noise z → fc to 8·nf·4×4 → stack of G_Block(up-sample) layers (get_G_in_out_chs) → to_rgb → Tanh.
    - Text conditioning via DFBLK + Affine: concatenates [z, sent_emb] and modulates feature maps in each block.

  - Discriminator:
    - NetD extracts multi-scale features with D_Block downsamplers.
    - NetC concatenates image features with sentence embedding (spatially replicated) to produce a conditional real/fake logit.
    - Losses use hinge; includes mismatched text negatives.
   - Regularization: Matching-Aware Gradient Penalty on image/text gradients.
   - Stabilization: Exponential Moving Average of G, EMA weights are used for testing/FID.

  - Evaluation:
    - FID: InceptionV3 2048-d features vs dataset npz.
    - CLIP alignment (optional diagnostic): generates a grid and cosine scores for prompts using ViT-B/32 CLIP pre-trained model.

- Standalone repo: https://github.com/The-GANners/DF-GAN

### Module 2 — CLIP-based Image Prompt Evaluator (Standalone Python)
- What it is:
  - A Python module that evaluates image–prompt alignment using CLIP ViT-B/32, with enhanced keyword analysis, contradiction checks, and score normalization.

- Inputs:
  - image_path: path to a PNG/JPEG/WebP image.
  - prompt: natural-language text.
  

- Outputs (dict):
  - overall_score, original_score, percentage_match, original_percentage, raw_percentage
  - quality (Excellent/Good/Fair/Poor), feedback, prompt
  - keyword_analysis: [{ keyword, present, confidence, raw_score, status, status_type }]
  - meets_threshold: bool
  - detailed_metrics: { raw_score, penalized_score, average_score, normalized_score, percentage, all_scores }
  - contradiction_warning (if penalty applied)
  - missing_feature_analysis: { present_features|weak_features|missing_features (with importance_weight, confidence, raw_score), counts }
  - missing_feature_feedback, missing_feature_penalty, penalty_percentage

- How it works (from code):
  - CLIP ViT-B/32 encodes image and multiple prompt variants (“a photo of …”, “an image showing …”, cleaned prompt); uses best similarity.
  - Applies semantic contradiction penalties for conflicting concepts if contradiction score exceeds the main score by a margin.
  - Extracts important keywords via NLTK (POS tagging, stopwords), boosts animals/living things using WordNet, and assigns importance weights.
  - Probes each keyword with multiple templates to classify present/weak/missing.
  - Computes a weighted missing-feature penalty; combines with contradiction penalty and recalculates normalized percentage.
  - Normalizes raw CLIP scores to 0–100% with calibrated bands and maps to quality:
    - Excellent > 0.28, Good > 0.22, Fair > 0.18, else Poor.

- Standalone repo: https://github.com/The-GANners/Module-2
- Core CLIP model: https://github.com/openai/CLIP

### Module 3 — Watermark UI (Standalone Desktop App)
- What it is:
  - A standalone Tkinter desktop application for batch watermarking images. The app provides a dark-themed UI, progress tracking, and safe overwrite behavior. 

- Inputs (from the UI):
  - Images: selected via folder picker or multiple file selection.
  - Watermark image: chosen from disk.
  - Options:
    - Position: NW | NE | SW | SE
    - Padding: ((padX, unit), (padY, unit)) where unit is "px" or "%"
    - Opacity: 0–100% (applied as 0.0–1.0 alpha)
    - Auto-resize watermark: on/off
  - Output:
    - Output directory and file renaming (prefix/postfix/none) via OutputSelector.
    - Overwrite toggle is handled per run (confirmation if files exist).

- Outputs:
  - Watermarked images saved to the chosen output directory (same extension as inputs), with optional prefix/suffix renaming. No network/API involved.

- How it works (from code):
  - App composition:
    - FileSelector: collects image paths; supports folder or multi-file selection.
    - OptionsPane: hosts WatermarkSelector, WatermarkOptions, and OutputSelector.
    - Worker: orchestrates batch processing, threads, progress bar, time estimation.
  - Processing loop:
    - Gathers files into a queue, asks for overwrite if needed, then spawns a worker thread.
    - For each image, builds output path.
    - Tracks progress/time via RemainingTime + Pacer.
  - Watermarking core:
      - Scaling: scales watermark based on input aspect (landscape/portrait/square) using configurable scale factors and min/max bounds.
      - Opacity: converts to RGBA and scales alpha channel per pixel if opacity < 1 (change_opacity).
      - Position: computes (x,y) from pos and padding; padding supports px/% units (get_watermark_position).
      - Caches resized watermark while processing a batch; respects overwrite flag.
- Standalone repo: https://github.com/The-GANners/Watermark-UI 

