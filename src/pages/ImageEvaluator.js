import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { 
  CheckCircleIcon, 
  CloudArrowUpIcon,
  ChartBarIcon,
  ExclamationTriangleIcon,
  StarIcon
} from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';

const ImageEvaluator = () => {
  const [image, setImage] = useState(null);
  const [prompt, setPrompt] = useState('');
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evaluation, setEvaluation] = useState(null);
  const [threshold, setThreshold] = useState(0.25);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.webp']
    },
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        const file = acceptedFiles[0];
        setImage({
          file,
          preview: URL.createObjectURL(file)
        });
      }
    },
    multiple: false
  });

  const handleEvaluate = async () => {
    if (!image || !prompt.trim()) {
      toast.error('Please upload an image and enter a prompt');
      return;
    }

    setIsEvaluating(true);
    try {
      // Simulate API call to evaluation backend
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Mock evaluation results
      const mockResults = {
        prompt: prompt,
        overallScore: Math.random() * 0.4 + 0.6, // 60-100%
        quality: '',
        feedback: '',
        keywordAnalysis: [
          { keyword: 'bird', present: true, confidence: 0.89 },
          { keyword: 'yellow', present: Math.random() > 0.5, confidence: Math.random() * 0.3 + 0.6 },
          { keyword: 'branch', present: Math.random() > 0.3, confidence: Math.random() * 0.4 + 0.5 },
          { keyword: 'small', present: Math.random() > 0.4, confidence: Math.random() * 0.3 + 0.4 }
        ],
        contradictions: [],
        suggestions: []
      };

      // Determine quality based on score
      if (mockResults.overallScore >= 0.8) {
        mockResults.quality = 'Excellent';
        mockResults.feedback = 'The image matches the prompt very well with high semantic similarity.';
      } else if (mockResults.overallScore >= 0.6) {
        mockResults.quality = 'Good';
        mockResults.feedback = 'The image shows good alignment with the prompt description.';
      } else if (mockResults.overallScore >= 0.4) {
        mockResults.quality = 'Fair';
        mockResults.feedback = 'The image has some elements matching the prompt but could be improved.';
      } else {
        mockResults.quality = 'Poor';
        mockResults.feedback = 'The image does not match the prompt well and needs significant improvement.';
      }

      setEvaluation({
        ...mockResults,
        percentageMatch: Math.round(mockResults.overallScore * 100)
      });

      toast.success('Evaluation completed successfully!');
    } catch (error) {
      toast.error('Failed to evaluate image. Please try again.');
    } finally {
      setIsEvaluating(false);
    }
  };

  const getQualityColor = (quality) => {
    switch (quality) {
      case 'Excellent': return 'text-green-600 bg-green-100';
      case 'Good': return 'text-blue-600 bg-blue-100';
      case 'Fair': return 'text-yellow-600 bg-yellow-100';
      case 'Poor': return 'text-red-600 bg-red-100';
      default: return 'text-gray-600 bg-gray-100';
    }
  };

  const getScoreColor = (score) => {
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-blue-600';
    if (score >= 40) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Image Quality Evaluation
          </h1>
          <p className="text-gray-600 text-lg max-w-2xl mx-auto">
            Analyze how well your AI-generated images match their text prompts using 
            advanced CLIP-based semantic understanding.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Input Panel */}
          <div className="space-y-6">
            {/* Image Upload */}
            <div className="card p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center space-x-2">
                <CloudArrowUpIcon className="w-5 h-5 text-primary-600" />
                <span>Upload Image</span>
              </h2>
              
              <div
                {...getRootProps()}
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer ${
                  isDragActive 
                    ? 'border-primary-400 bg-primary-50' 
                    : 'border-gray-300 hover:border-primary-400'
                }`}
              >
                <input {...getInputProps()} />
                {image ? (
                  <div className="space-y-4">
                    <img 
                      src={image.preview} 
                      alt="Uploaded" 
                      className="max-h-64 mx-auto rounded-lg shadow-sm"
                    />
                    <p className="text-sm text-gray-600">
                      Click or drag to replace image
                    </p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <CloudArrowUpIcon className="w-12 h-12 text-gray-400 mx-auto" />
                    <div>
                      <p className="text-lg text-gray-600 mb-2">
                        {isDragActive ? 'Drop your image here' : 'Drag & drop an image here'}
                      </p>
                      <p className="text-sm text-gray-400">
                        or click to select • PNG, JPG, JPEG, WebP
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Prompt Input */}
            <div className="card p-6">
              <h2 className="text-xl font-semibold mb-4">Text Prompt</h2>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Enter the text prompt used to generate this image..."
                className="textarea h-32 mb-4"
                disabled={isEvaluating}
              />
              
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Similarity Threshold: {threshold}
                </label>
                <input
                  type="range"
                  min="0.1"
                  max="0.4"
                  step="0.01"
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  className="w-full"
                  disabled={isEvaluating}
                />
                <p className="text-xs text-gray-500 mt-1">
                  Lower values are more lenient, higher values are stricter
                </p>
              </div>

              <button
                onClick={handleEvaluate}
                disabled={isEvaluating || !image || !prompt.trim()}
                className="btn-primary w-full flex items-center justify-center space-x-2"
              >
                {isEvaluating ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span>Evaluating...</span>
                  </>
                ) : (
                  <>
                    <CheckCircleIcon className="w-5 h-5" />
                    <span>Evaluate Image</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Results Panel */}
          <div className="space-y-6">
            {evaluation ? (
              <>
                {/* Overall Score */}
                <div className="card p-6">
                  <h2 className="text-xl font-semibold mb-4 flex items-center space-x-2">
                    <ChartBarIcon className="w-5 h-5 text-primary-600" />
                    <span>Evaluation Results</span>
                  </h2>

                  <div className="text-center mb-6">
                    <div className={`text-6xl font-bold mb-2 ${getScoreColor(evaluation.percentageMatch)}`}>
                      {evaluation.percentageMatch}%
                    </div>
                    <div className={`inline-flex px-3 py-1 rounded-full text-sm font-medium ${getQualityColor(evaluation.quality)}`}>
                      {evaluation.quality}
                    </div>
                  </div>

                  <p className="text-gray-600 text-center mb-6">
                    {evaluation.feedback}
                  </p>

                  {/* Progress Bar */}
                  <div className="w-full bg-gray-200 rounded-full h-3 mb-4">
                    <div 
                      className={`h-3 rounded-full progress-bar ${
                        evaluation.percentageMatch >= 80 ? 'bg-green-500' :
                        evaluation.percentageMatch >= 60 ? 'bg-blue-500' :
                        evaluation.percentageMatch >= 40 ? 'bg-yellow-500' : 'bg-red-500'
                      }`}
                      style={{ width: `${evaluation.percentageMatch}%` }}
                    />
                  </div>
                </div>

                {/* Keyword Analysis */}
                <div className="card p-6">
                  <h3 className="text-lg font-semibold mb-4">Keyword Analysis</h3>
                  <div className="space-y-3">
                    {evaluation.keywordAnalysis.map((keyword, index) => (
                      <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                        <div className="flex items-center space-x-3">
                          {keyword.present ? (
                            <CheckCircleIcon className="w-5 h-5 text-green-500" />
                          ) : (
                            <ExclamationTriangleIcon className="w-5 h-5 text-red-500" />
                          )}
                          <span className="font-medium">{keyword.keyword}</span>
                        </div>
                        <div className="flex items-center space-x-2">
                          <div className="flex items-center">
                            {[...Array(5)].map((_, i) => (
                              <StarIcon 
                                key={i}
                                className={`w-4 h-4 ${
                                  i < Math.round(keyword.confidence * 5) 
                                    ? 'text-yellow-400 fill-current' 
                                    : 'text-gray-300'
                                }`}
                              />
                            ))}
                          </div>
                          <span className="text-sm text-gray-600">
                            {Math.round(keyword.confidence * 100)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            ) : (
              <div className="card p-6 text-center text-gray-500">
                <ChartBarIcon className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                <p>Upload an image and enter a prompt to see evaluation results</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImageEvaluator;
