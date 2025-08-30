# AI Image Processing Suite

A comprehensive web application that integrates three powerful AI image processing modules:

1. **DF-GAN Text-to-Image Generation (Module 1)** - Generate stunning images from text descriptions
2. **Image Prompt Evaluator (Module 2)** - Evaluate how well AI-generated images match their text prompts
3. **Watermark Protection UI (Module 3)** - Add professional watermarks to protect your images

## Features

### 🎨 Text-to-Image Generation
- Generate images from text descriptions using the DF-GAN model
- Support for multiple datasets (Birds, COCO, Flowers)
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
- Dark/light mode support
- Drag-and-drop file uploads
- Real-time progress indicators
- Toast notifications for user feedback
- Image gallery with filtering and search

## Technology Stack

- **Frontend**: React 18 with functional components and hooks
- **Styling**: Tailwind CSS with custom components
- **Icons**: Heroicons
- **Routing**: React Router DOM
- **File Handling**: React Dropzone
- **Notifications**: React Hot Toast
- **Animations**: Framer Motion
- **Build Tool**: Create React App

## Server Setup and Path Configuration

The AI-Image-Suite server uses a flexible path configuration system that works across different development environments and systems.

### Quick Setup

1. **Clone the repositories**
   ```bash
   # Clone the main project
   git clone <ai-image-suite-repo>
   cd AI-Image-Suite
   
   # Clone DF-GAN in the parent directory
   cd ..
   git clone <df-gan-repo>
   ```

2. **Install Python dependencies**
   ```bash
   cd AI-Image-Suite
   pip install -r requirements.txt
   ```

3. **Configure paths (optional)**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env if your paths differ from defaults
   ```

### Path Configuration

The server automatically detects paths based on the project structure. The default layout expected is:

```
your-projects-folder/
├── AI-Image-Suite/          # This repository
│   ├── server/
│   │   ├── app.py
│   │   ├── config.py
│   │   └── dfgan_wrapper.py
│   └── models/
│       ├── CUB.pth
│       └── COCO.pth
└── DF-GAN/                  # DF-GAN repository
    ├── code/
    ├── data/
    └── models/
```

### Environment Variables

If your setup differs from the default structure, you can override paths using environment variables:

```bash
# .env file
DF_GAN_PATH=/path/to/DF-GAN
AI_SUITE_ROOT=/path/to/AI-Image-Suite
CUB_WEIGHTS=/path/to/CUB.pth
COCO_WEIGHTS=/path/to/COCO.pth
PORT=5001
```

### Cross-Platform Compatibility

The configuration system automatically handles:
- **Windows paths**: `D:\Projects\FYP\DF-GAN`
- **Linux/Mac paths**: `/home/user/projects/DF-GAN`
- **Relative paths**: `../DF-GAN`
- **Environment variables**: `$DF_GAN_PATH`

### Validation

Check if your setup is correct:

```bash
# Start the server
python server/app.py

# Check the setup endpoint
curl http://localhost:5001/api/check
```

The `/api/check` endpoint will report any missing files or incorrect paths.

### Version Control

The following files are automatically ignored by git:
- `.env` (local environment configuration)
- `myenv/` (Python virtual environments)
- Temporary output directories

This ensures that system-specific paths are not committed to version control.

## Project Structure

```
src/
├── components/
│   └── Navbar.js          # Navigation component
├── pages/
│   ├── Dashboard.js       # Homepage with overview
│   ├── TextToImage.js     # DF-GAN image generation
│   ├── ImageEvaluator.js  # CLIP-based evaluation
│   ├── Watermark.js       # Watermarking interface
│   └── Gallery.js         # Image management
├── App.js                 # Main app component
├── index.js              # Entry point
└── index.css             # Global styles
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AI-Image-Suite
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm start
   ```

4. **Open your browser**
   Navigate to `http://localhost:3000`

## Backend Integration

This frontend is designed to work with three backend modules:

### Module 1: DF-GAN Backend
- **Endpoint**: `/api/generate`
- **Method**: POST
- **Payload**: 
  ```json
  {
    "prompt": "string",
    "dataset": "bird|coco|flower",
    "batch_size": 1-4,
    "steps": 20-200,
    "guidance": 1.0-20.0,
    "seed": -1 or positive integer
  }
  ```

### Module 2: Image Evaluator Backend
- **Endpoint**: `/api/evaluate`
- **Method**: POST (multipart/form-data)
- **Payload**:
  ```
  image: File
  prompt: string
  threshold: 0.1-0.4
  ```

### Module 3: Watermark Backend
- **Endpoint**: `/api/watermark`
- **Method**: POST (multipart/form-data)
- **Payload**:
  ```
  images: File[]
  watermark: File
  settings: WatermarkSettings object
  ```

## Configuration

Create a `.env` file in the root directory:

```env
REACT_APP_API_BASE_URL=http://localhost:5000
REACT_APP_DFGAN_ENDPOINT=/api/generate
REACT_APP_EVALUATOR_ENDPOINT=/api/evaluate
REACT_APP_WATERMARK_ENDPOINT=/api/watermark
```

## Features in Detail

### Dashboard
- Overview of all three modules
- Usage statistics
- Quick access to main features
- Recent activity feed

### Text-to-Image Generation
- Multiple dataset support (CUB-200-2011 Birds, COCO, Oxford Flowers)
- Real-time generation with progress indicators
- Customizable parameters (steps, guidance, batch size, seed)
- Example prompts for each dataset
- Generated image gallery with metadata

### Image Evaluation
- Upload images via drag-and-drop
- CLIP model-based semantic analysis
- Keyword presence detection
- Confidence scoring for each element
- Quality ratings and feedback
- Adjustable similarity thresholds

### Watermarking
- Bulk image processing
- Multiple positioning options (corners, center)
- Opacity and scale controls
- Padding customization
- File naming patterns (prefix/suffix)
- Auto-resize functionality
- Batch download with ZIP compression

### Gallery
- Centralized image management
- Category filtering (Generated, Evaluated, Watermarked)
- Search by prompt or tags
- Sort by date or quality score
- Image actions (view, share, download, favorite)
- Metadata display

## Responsive Design

The application is fully responsive and works seamlessly across:
- Desktop (1024px+)
- Tablet (768px - 1023px)
- Mobile (320px - 767px)

## Performance Optimizations

- Lazy loading for images
- Optimized re-renders with React.memo
- Efficient state management
- Image compression and caching
- Progressive loading indicators

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [DF-GAN](https://github.com/tobran/DF-GAN) for text-to-image generation
- [CLIP](https://github.com/openai/CLIP) for image-text similarity evaluation
- [FreeMark](https://github.com/nikolajlauridsen/FreeMark) for watermarking inspiration
- [Heroicons](https://heroicons.com/) for beautiful icons
- [Tailwind CSS](https://tailwindcss.com/) for styling framework
