// API configuration
export const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000';

export const API_ENDPOINTS = {
  GENERATE: process.env.REACT_APP_DFGAN_ENDPOINT || '/api/generate',
  EVALUATE: process.env.REACT_APP_EVALUATOR_ENDPOINT || '/api/evaluate', 
  WATERMARK: process.env.REACT_APP_WATERMARK_ENDPOINT || '/api/watermark',
};

// API utility functions
export const apiCall = async (endpoint, options = {}) => {
  const url = `${API_BASE_URL}${endpoint}`;
  const config = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  };

  try {
    const response = await fetch(url, config);
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const contentType = response.headers.get('content-type');
    if (contentType && contentType.includes('application/json')) {
      return await response.json();
    }
    
    return response;
  } catch (error) {
    console.error('API call failed:', error);
    throw error;
  }
};

// Text-to-Image Generation API
export const generateImages = async (prompt, settings) => {
  const formData = new FormData();
  formData.append('prompt', prompt);
  formData.append('dataset', settings.dataset);
  formData.append('batch_size', settings.batchSize);
  formData.append('steps', settings.steps);
  formData.append('guidance', settings.guidance);
  formData.append('seed', settings.seed);

  return apiCall(API_ENDPOINTS.GENERATE, {
    method: 'POST',
    body: formData,
    headers: {}, // Remove Content-Type to let browser set it for FormData
  });
};

// Image Evaluation API
export const evaluateImage = async (imageFile, prompt, threshold) => {
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('prompt', prompt);
  formData.append('threshold', threshold);

  return apiCall(API_ENDPOINTS.EVALUATE, {
    method: 'POST',
    body: formData,
    headers: {}, // Remove Content-Type to let browser set it for FormData
  });
};

// Watermarking API
export const watermarkImages = async (images, watermark, settings) => {
  const formData = new FormData();
  
  // Add all images
  images.forEach((image, index) => {
    formData.append(`images`, image.file);
  });
  
  // Add watermark
  formData.append('watermark', watermark.file);
  
  // Add settings
  formData.append('settings', JSON.stringify(settings));

  return apiCall(API_ENDPOINTS.WATERMARK, {
    method: 'POST',
    body: formData,
    headers: {}, // Remove Content-Type to let browser set it for FormData
  });
};
