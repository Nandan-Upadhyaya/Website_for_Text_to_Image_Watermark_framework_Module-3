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
    console.log("🔍 [FRONTEND] Starting evaluation...");
    
    if (!image || !prompt.trim()) {
      toast.error('Please upload an image and enter a prompt');
      return;
    }

    setIsEvaluating(true);
    
    try {
      // Test API connectivity first
      console.log("🔍 [FRONTEND] Testing API connectivity...");
      try {
        const testResponse = await fetch('/api/test', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ test: 'connectivity' })
        });
        
        if (!testResponse.ok) {
          throw new Error(`API test failed: ${testResponse.status}`);
        }
        
        const testData = await testResponse.json();
        console.log("✅ [FRONTEND] API connectivity test passed:", testData);
      } catch (testError) {
        console.error("❌ [FRONTEND] API connectivity test failed:", testError);
        throw new Error(`Cannot reach backend API: ${testError.message}`);
      }

      // Convert image to base64
      console.log("🔍 [FRONTEND] Converting image to base64...");
      const base64Image = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(image.file);
      });

      const requestBody = {
        image: base64Image,
        prompt: prompt.trim(),
        threshold: threshold
      };

      console.log("🔍 [FRONTEND] Request body size:", JSON.stringify(requestBody).length);
      console.log("🔍 [FRONTEND] Calling /api/evaluate-image...");
      
      // Call evaluation API with detailed error handling
      const response = await fetch('/api/evaluate-image', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });

      console.log(`🔍 [FRONTEND] Response status: ${response.status}`);
      console.log(`🔍 [FRONTEND] Response headers:`, Object.fromEntries(response.headers));

      // Check if response is actually JSON
      const contentType = response.headers.get('content-type');
      console.log(`🔍 [FRONTEND] Content-Type: ${contentType}`);
      
      if (!contentType || !contentType.includes('application/json')) {
        // Response is not JSON - likely an error page
        const textResponse = await response.text();
        console.error("❌ [FRONTEND] Non-JSON response received:", textResponse.substring(0, 500));
        throw new Error(`Server returned HTML instead of JSON. Check if backend is running on port 5001.`);
      }

      const results = await response.json();
      console.log("✅ [FRONTEND] Got JSON response:", Object.keys(results));

      if (!response.ok) {
        throw new Error(results.error || `HTTP ${response.status}`);
      }

      if (results.error) {
        throw new Error(results.error);
      }

      // Check for required fields
      if (!results.percentage_match || !results.quality) {
        console.error("❌ [FRONTEND] Invalid response structure:", results);
        throw new Error('Invalid response from server - missing required fields');
      }

      console.log("✅ [FRONTEND] Valid evaluation results received");

      // Transform Module-2 results for frontend display
      const transformedResults = {
        prompt: results.prompt,
        overallScore: results.overall_score,
        percentageMatch: parseFloat(results.percentage_match.replace('%', '')),
        quality: results.quality,
        feedback: results.feedback,
        keywordAnalysis: (results.keyword_analysis || []).map(kw => ({
          keyword: kw.keyword,
          present: kw.status_type === 'present',
          confidence: parseFloat(kw.confidence.replace('%', '')) / 100,
          confidencePercent: kw.confidence,
          status: kw.status,
          statusType: kw.status_type,
          rawScore: kw.raw_score || 0
        })),
        contradictionWarning: results.contradiction_warning,
        missingFeatureAnalysis: results.missing_feature_analysis,
        detailedMetrics: results.detailed_metrics || { raw_score: 0, average_score: 0 }
      };

      console.log("✅ [FRONTEND] Results transformed successfully");
      setEvaluation(transformedResults);
      toast.success('Evaluation completed successfully!');
      
    } catch (error) {
      console.error("❌ [FRONTEND] Evaluation failed:", error);
      toast.error(`Failed to evaluate: ${error.message}`);
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

          {/* Results Panel - Updated to show real Module-2 results */}
          <div className="space-y-6">
            {evaluation ? (
              <>
                {/* Overall Score */}
                <div className="card p-6">
                  <h2 className="text-xl font-semibold mb-4 flex items-center space-x-2">
                    <ChartBarIcon className="w-5 h-5 text-primary-600" />
                    <span>Module-2 Evaluation Results</span>
                  </h2>

                  <div className="text-center mb-6">
                    <div className={`text-6xl font-bold mb-2 ${getScoreColor(evaluation.percentageMatch)}`}>
                      {evaluation.percentageMatch.toFixed(2)}%
                    </div>
                    <div className={`inline-flex px-3 py-1 rounded-full text-sm font-medium ${getQualityColor(evaluation.quality)}`}>
                      {evaluation.quality}
                    </div>
                  </div>

                  <p className="text-gray-600 text-center mb-6">
                    {evaluation.feedback}
                  </p>

                  {/* Contradiction Warning */}
                  {evaluation.contradictionWarning && (
                    <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg text-sm text-yellow-800">
                      <ExclamationTriangleIcon className="w-4 h-4 inline mr-2" />
                      {evaluation.contradictionWarning}
                    </div>
                  )}

                  {/* Progress Bar */}
                  <div className="w-full bg-gray-200 rounded-full h-3 mb-4">
                    <div 
                      className={`h-3 rounded-full progress-bar ${
                        evaluation.percentageMatch >= 80 ? 'bg-green-500' :
                        evaluation.percentageMatch >= 60 ? 'bg-blue-500' :
                        evaluation.percentageMatch >= 40 ? 'bg-yellow-500' : 'bg-red-500'
                      }`}
                      style={{ width: `${Math.min(100, evaluation.percentageMatch)}%` }}
                    />
                  </div>

                  {/* Detailed Metrics */}
                  {evaluation.detailedMetrics && (
                    <div className="mt-4 text-xs text-gray-500 space-y-1">
                      <div className="flex justify-between">
                        <span>Raw CLIP Score:</span>
                        <span className="font-mono">{evaluation.detailedMetrics.raw_score?.toFixed(4)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Average Score:</span>
                        <span className="font-mono">{evaluation.detailedMetrics.average_score?.toFixed(4)}</span>
                      </div>
                    </div>
                  )}
                </div>

                {/* Keyword Analysis - Real Module-2 Data */}
                <div className="card p-6">
                  <h3 className="text-lg font-semibold mb-4">Detailed Keyword Analysis</h3>
                  <div className="space-y-3">
                    {evaluation.keywordAnalysis.map((keyword, index) => (
                      <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                        <div className="flex items-center space-x-3">
                          {keyword.statusType === 'present' ? (
                            <CheckCircleIcon className="w-5 h-5 text-green-500" />
                          ) : keyword.statusType === 'weak' ? (
                            <ExclamationTriangleIcon className="w-5 h-5 text-yellow-500" />
                          ) : (
                            <ExclamationTriangleIcon className="w-5 h-5 text-red-500" />
                          )}
                          <div>
                            <span className="font-medium">{keyword.keyword}</span>
                            <span className="ml-2 text-xs text-gray-500">{keyword.status}</span>
                          </div>
                        </div>
                        <div className="flex items-center space-x-2">
                          <span className="text-sm font-medium text-gray-700">
                            {keyword.confidencePercent}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Feature Analysis Summary */}
                {evaluation.missingFeatureAnalysis && (
                  <div className="card p-6">
                    <h3 className="text-lg font-semibold mb-4">Feature Analysis</h3>
                    
                    {evaluation.missingFeatureAnalysis.present_features?.length > 0 && (
                      <div className="mb-4">
                        <h4 className="text-sm font-semibold text-green-700 mb-2">
                          ✅ Present Features ({evaluation.missingFeatureAnalysis.present_features.length})
                        </h4>
                        <div className="flex flex-wrap gap-2">
                          {evaluation.missingFeatureAnalysis.present_features.map((f, i) => (
                            <span key={i} className="px-2 py-1 bg-green-100 text-green-800 rounded text-xs">
                              {f.keyword} ({f.confidence?.toFixed(1)}%)
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {evaluation.missingFeatureAnalysis.weak_features?.length > 0 && (
                      <div className="mb-4">
                        <h4 className="text-sm font-semibold text-yellow-700 mb-2">
                          ⚠️ Weak Features ({evaluation.missingFeatureAnalysis.weak_features.length})
                        </h4>
                        <div className="flex flex-wrap gap-2">
                          {evaluation.missingFeatureAnalysis.weak_features.map((f, i) => (
                            <span key={i} className="px-2 py-1 bg-yellow-100 text-yellow-800 rounded text-xs">
                              {f.keyword} ({f.confidence?.toFixed(1)}%)
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {evaluation.missingFeatureAnalysis.missing_features?.length > 0 && (
                      <div>
                        <h4 className="text-sm font-semibold text-red-700 mb-2">
                          ❌ Missing Features ({evaluation.missingFeatureAnalysis.missing_features.length})
                        </h4>
                        <div className="flex flex-wrap gap-2">
                          {evaluation.missingFeatureAnalysis.missing_features.map((f, i) => (
                            <span key={i} className="px-2 py-1 bg-red-100 text-red-800 rounded text-xs">
                              {f.keyword} ({f.confidence?.toFixed(1)}%)
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </>
            ) : (
              <div className="card p-6 text-center text-gray-500">
                <ChartBarIcon className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                <p>Upload an image and enter a prompt to see evaluation results</p>
                <p className="text-xs mt-2">Using Module-2 CLIP-based evaluation</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ImageEvaluator;
