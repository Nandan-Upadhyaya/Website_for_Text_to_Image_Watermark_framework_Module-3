// Constants for the AI Image Suite

// API Configuration
export const API_CONFIG = {
  TIMEOUT: 30000, // 30 seconds
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000, // 1 second
};

// File Upload Limits
export const FILE_LIMITS = {
  MAX_IMAGE_SIZE: 10 * 1024 * 1024, // 10MB
  MAX_BATCH_SIZE: 10,
  SUPPORTED_IMAGE_TYPES: [
    'image/jpeg',
    'image/jpg', 
    'image/png',
    'image/webp'
  ],
  SUPPORTED_EXTENSIONS: ['.jpg', '.jpeg', '.png', '.webp'],
};

// DF-GAN Configuration
export const DFGAN_CONFIG = {
  DATASETS: [
    { value: 'bird', label: 'Birds (CUB-200-2011)', description: 'High-quality bird images' },
    { value: 'coco', label: 'COCO Dataset', description: 'General objects and scenes' },
    { value: 'flower', label: 'Oxford Flowers', description: 'Beautiful flower varieties' }
  ],
  DEFAULT_SETTINGS: {
    dataset: 'bird',
    batchSize: 1,
    steps: 100,
    guidance: 7.5,
    seed: -1
  },
  LIMITS: {
    MIN_BATCH_SIZE: 1,
    MAX_BATCH_SIZE: 4,
    MIN_STEPS: 20,
    MAX_STEPS: 200,
    MIN_GUIDANCE: 1.0,
    MAX_GUIDANCE: 20.0,
    MIN_SEED: -1,
    MAX_SEED: 2147483647
  }
};

// Example Prompts
export const EXAMPLE_PROMPTS = {
  bird: [
    "A small yellow bird with black wings perched on a branch",
    "A red cardinal sitting on a snowy pine tree",
    "A blue jay with distinctive crest feathers",
    "A tiny hummingbird hovering near purple flowers",
    "A majestic eagle soaring through cloudy skies",
    "A colorful parrot with bright green and red feathers"
  ],
  coco: [
    "A cat sitting on a wooden table in a sunny kitchen",
    "A dog playing fetch in a green park",
    "A bicycle leaning against a brick wall",
    "Children playing soccer in a field",
    "A vintage car parked on a cobblestone street",
    "People enjoying a picnic by the lake"
  ],
  flower: [
    "A single red rose with morning dew drops",
    "A field of sunflowers facing the sun",
    "Purple lavender swaying in the breeze",
    "A bouquet of colorful tulips in spring",
    "White daisies growing in a meadow",
    "Cherry blossoms on a tree branch"
  ]
};

// Image Evaluator Configuration
export const EVALUATOR_CONFIG = {
  DEFAULT_THRESHOLD: 0.25,
  MIN_THRESHOLD: 0.1,
  MAX_THRESHOLD: 0.4,
  QUALITY_THRESHOLDS: {
    EXCELLENT: 0.8,
    GOOD: 0.6,
    FAIR: 0.4,
    POOR: 0.0
  },
  QUALITY_LABELS: {
    EXCELLENT: 'Excellent',
    GOOD: 'Good', 
    FAIR: 'Fair',
    POOR: 'Poor'
  }
};

// Watermark Configuration
export const WATERMARK_CONFIG = {
  POSITIONS: [
    { value: 'top-left', label: 'Top Left' },
    { value: 'top-right', label: 'Top Right' },
    { value: 'bottom-left', label: 'Bottom Left' },
    { value: 'bottom-right', label: 'Bottom Right' },
    { value: 'center', label: 'Center' }
  ],
  DEFAULT_SETTINGS: {
    position: 'bottom-right',
    opacity: 80,
    scale: 20,
    paddingX: 5,
    paddingY: 5,
    paddingUnit: 'percentage',
    prefix: '',
    suffix: '',
    autoResize: true
  },
  LIMITS: {
    MIN_OPACITY: 10,
    MAX_OPACITY: 100,
    MIN_SCALE: 5,
    MAX_SCALE: 50,
    MIN_PADDING: 0,
    MAX_PADDING: 20
  }
};

// Gallery Configuration
export const GALLERY_CONFIG = {
  CATEGORIES: [
    { value: 'all', label: 'All Images' },
    { value: 'generated', label: 'Generated' },
    { value: 'evaluated', label: 'Evaluated' },
    { value: 'watermarked', label: 'Watermarked' }
  ],
  SORT_OPTIONS: [
    { value: 'newest', label: 'Newest First' },
    { value: 'oldest', label: 'Oldest First' },
    { value: 'highest-score', label: 'Highest Score' },
    { value: 'lowest-score', label: 'Lowest Score' }
  ],
  ITEMS_PER_PAGE: 12
};

// UI Constants
export const UI_CONSTANTS = {
  ANIMATION_DURATION: 300,
  DEBOUNCE_DELAY: 300,
  TOAST_DURATION: 4000,
  MOBILE_BREAKPOINT: 768,
  TABLET_BREAKPOINT: 1024,
};

// Local Storage Keys
export const STORAGE_KEYS = {
  USER_PREFERENCES: 'ai_image_suite_preferences',
  GENERATED_IMAGES: 'ai_image_suite_generated_images',
  EVALUATION_HISTORY: 'ai_image_suite_evaluation_history',
  WATERMARK_SETTINGS: 'ai_image_suite_watermark_settings',
  RECENT_PROMPTS: 'ai_image_suite_recent_prompts'
};

// Color Schemes
export const COLOR_SCHEMES = {
  PRIMARY: {
    50: '#eff6ff',
    100: '#dbeafe', 
    500: '#3b82f6',
    600: '#2563eb',
    700: '#1d4ed8'
  },
  SUCCESS: {
    500: '#10b981',
    600: '#059669'
  },
  WARNING: {
    500: '#f59e0b',
    600: '#d97706'
  },
  ERROR: {
    500: '#ef4444',
    600: '#dc2626'
  }
};

// Status Messages
export const STATUS_MESSAGES = {
  GENERATING: 'Generating images...',
  EVALUATING: 'Evaluating image quality...',
  WATERMARKING: 'Applying watermarks...',
  UPLOADING: 'Uploading files...',
  PROCESSING: 'Processing...',
  SUCCESS: 'Operation completed successfully!',
  ERROR: 'An error occurred. Please try again.',
  NETWORK_ERROR: 'Network error. Please check your connection.',
  FILE_TOO_LARGE: 'File size exceeds the limit.',
  INVALID_FILE_TYPE: 'Invalid file type. Please select a supported image format.',
  NO_FILES_SELECTED: 'Please select files to process.',
  EMPTY_PROMPT: 'Please enter a text prompt.'
};

// Feature Flags
export const FEATURE_FLAGS = {
  ENABLE_BATCH_GENERATION: true,
  ENABLE_ADVANCED_SETTINGS: true,
  ENABLE_IMAGE_PREVIEW: true,
  ENABLE_HISTORY: true,
  ENABLE_EXPORT: true,
  ENABLE_SHARING: false // Not implemented in frontend-only version
};
