import React, { useState, useEffect } from 'react';
import { PhotoIcon, SparklesIcon, AdjustmentsHorizontalIcon, XMarkIcon } from '@heroicons/react/24/outline';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';

const TextToImage = () => {
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImages, setGeneratedImages] = useState([]);
  const [serverStatus, setServerStatus] = useState({ status: 'unknown', message: 'Checking server...' });
  const [settings, setSettings] = useState({
    model: 'df-gan',
    dataset: 'bird',
    batchSize: 1,
    seed: -1
  });
  const [showEvaluationDialog, setShowEvaluationDialog] = useState(false);
  const [pendingEvaluationData, setPendingEvaluationData] = useState(null);
  const navigate = useNavigate();

  const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:5001';

  const datasets = [
    { value: 'bird', label: 'Birds (CUB-200-2011)' },
    { value: 'coco', label: 'COCO Dataset' },
    
  ];

  const examplePrompts = {
    bird: [
      "this bird is white with brown and has a very short beak.",
      "this bird has an orange bill, a white belly and white eyebrows.",
      "A blue jay with distinctive crest feathers",
      "this bird is white with red and has a very short beak."
    ],
    coco: [
      "A boat in the middle of the ocean.",
      "A large construction site for a bridge build.  ",
      "A kitchen has white counters and a wooden floor.",
      "On the plate is eggs,tomatoes sausage, and some bacon."
    ],
  };

  // Check server status on component mount
  useEffect(() => {
    const checkServer = async () => {
      try {
        const res = await fetch(`${API_BASE}/api/check`);
        const data = await res.json();
        
        if (res.ok) {
          setServerStatus({
            status: 'ready',
            message: 'Server is ready'
          });
        } else {
          setServerStatus({
            status: 'error',
            message: `Server issues: ${data.issues?.join(', ')}`,
            details: data
          });
        }
      } catch (error) {
        setServerStatus({
          status: 'error',
          message: 'Cannot connect to server. Is it running?'
        });
      }
    };
    
    checkServer();
  }, [API_BASE]);

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      toast.error('Please enter a text prompt');
      return;
    }

    setIsGenerating(true);
    try {
      // Generate a unique seed for each image in the batch
      const seeds = Array.from({ length: settings.batchSize }, () => Math.floor(Math.random() * 1000000));
      const res = await fetch(`${API_BASE}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: prompt.trim(),
          dataset: settings.dataset, // 'bird' -> CUB, 'coco' -> COCO
          batchSize: settings.batchSize,
          seeds: seeds
        })
      });

      if (!res.ok) {
        // Prefer JSON error if provided by backend
        let msg = 'Generation failed';
        try {
          const j = await res.json();
          msg = j?.error || msg;
        } catch {
          const text = await res.text();
          msg = text || msg;
        }
        throw new Error(msg);
      }

      const data = await res.json();
      const images = (data.images || []).map((b64, i) => ({
        id: Date.now() + i,
        url: `data:image/png;base64,${b64}`,
        prompt,
        settings: { ...settings, seed: seeds[i] },
        createdAt: new Date()
      }));

      if (!images.length) throw new Error('No images returned from DF-GAN');

      setGeneratedImages(prev => [...images, ...prev]);
      toast.success(`Generated ${images.length} image(s) successfully!`);
      
      // NEW: Trigger auto-evaluation suggestion after 1.5 seconds
      setTimeout(() => {
        setPendingEvaluationData({
          prompt: prompt.trim(),
          images: images,
          timestamp: Date.now()
        });
        setShowEvaluationDialog(true);
      }, 1500);
      
    } catch (error) {
      toast.error(error.message || 'Failed to generate images. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  };

  // NEW: Handle auto-evaluation acceptance
  const handleAutoEvaluationAccept = () => {
    if (pendingEvaluationData) {
      // Store the evaluation data globally for the ImageEvaluator to pick up
      sessionStorage.setItem('autoEvaluationData', JSON.stringify({
        prompt: pendingEvaluationData.prompt,
        images: pendingEvaluationData.images.map(img => ({
          id: img.id,
          url: img.url,
          prompt: img.prompt
        })),
        timestamp: pendingEvaluationData.timestamp
      }));
      
      // Navigate to evaluation page
      navigate('/evaluate');
    }
    setShowEvaluationDialog(false);
    setPendingEvaluationData(null);
  };

  // NEW: Handle auto-evaluation decline
  const handleAutoEvaluationDecline = () => {
    setShowEvaluationDialog(false);
    setPendingEvaluationData(null);
  };

  const setExamplePrompt = (examplePrompt) => {
    setPrompt(examplePrompt);
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Text-to-Image Generation
          </h1>
          <p className="text-gray-600 text-lg max-w-2xl mx-auto">
            Create stunning images from text descriptions using the DF-GAN model.
            Enter your prompt and customize the generation settings below.
          </p>
          
          {/* Server Status Indicator */}
          <div className={`mt-4 inline-flex items-center px-4 py-2 rounded-full text-sm ${
            serverStatus.status === 'ready' 
              ? 'bg-green-100 text-green-800' 
              : serverStatus.status === 'error'
                ? 'bg-red-100 text-red-800'
                : 'bg-yellow-100 text-yellow-800'
          }`}>
            <div className={`w-2 h-2 rounded-full mr-2 ${
              serverStatus.status === 'ready' 
                ? 'bg-green-500' 
                : serverStatus.status === 'error'
                  ? 'bg-red-500'
                  : 'bg-yellow-500'
            }`}></div>
            {serverStatus.message}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Input Panel */}
          <div className="lg:col-span-2 space-y-6">
            {/* Prompt Input */}
            <div className="card p-6">
              <div className="flex items-center space-x-2 mb-4">
                <SparklesIcon className="w-5 h-5 text-primary-600" />
                <h2 className="text-xl font-semibold">Prompt</h2>
              </div>
              
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="Describe the image you want to generate..."
                className="textarea h-32 mb-4"
                disabled={isGenerating}
              />

              {/* Example Prompts */}
              <div className="mb-4">
                <h3 className="text-sm font-medium text-gray-700 mb-2">
                  Example prompts for {datasets.find(d => d.value === settings.dataset)?.label}:
                </h3>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {examplePrompts[settings.dataset]?.map((example, index) => (
                    <button
                      key={index}
                      onClick={() => setExamplePrompt(example)}
                      className="text-left p-2 text-sm bg-gray-50 hover:bg-primary-50 hover:text-primary-700 rounded border text-gray-600 transition-colors"
                      disabled={isGenerating}
                    >
                      "{example}"
                    </button>
                  ))}
                </div>
              </div>

              <button
                onClick={handleGenerate}
                disabled={isGenerating || !prompt.trim()}
                className="btn-primary w-full flex items-center justify-center space-x-2"
              >
                {isGenerating ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span className="loading-dots">Generating</span>
                  </>
                ) : (
                  <>
                    <PhotoIcon className="w-5 h-5" />
                    <span>Generate Images</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Settings Panel */}
          <div className="space-y-6">
            <div className="card p-6">
              <div className="flex items-center space-x-2 mb-4">
                <AdjustmentsHorizontalIcon className="w-5 h-5 text-primary-600" />
                <h2 className="text-xl font-semibold">Settings</h2>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Dataset
                  </label>
                  <select
                    value={settings.dataset}
                    onChange={(e) => setSettings({...settings, dataset: e.target.value})}
                    className="input"
                    disabled={isGenerating}
                  >
                    {datasets.map(dataset => (
                      <option key={dataset.value} value={dataset.value}>
                        {dataset.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Number of Images
                  </label>
                  <input
                    type="number"
                    min="1"
                    max="20"
                    value={settings.batchSize}
                    onChange={(e) => setSettings({...settings, batchSize: Math.max(1, Math.min(20, parseInt(e.target.value) || 1))})}
                    className="input"
                    disabled={isGenerating}
                  />
                </div>

                <div>
                  {/* Seed input removed: seeds are now auto-generated per image */}
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Generated Images */}
        {generatedImages.length > 0 && (
          <div className="mt-8">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-900">Generated Images</h2>
              <button
                className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600 transition"
                onClick={() => setGeneratedImages([])}
              >
                Clear Images
              </button>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {generatedImages.map((image) => (
                <div key={image.id} className="card p-4 image-fade-in">
                  <img
                    src={image.url}
                    alt={image.prompt}
                    className="w-full aspect-square object-cover rounded-lg mb-3"
                  />
                  <p className="text-sm text-gray-600 line-clamp-2 mb-2">
                    "{image.prompt}"
                  </p>
                  <div className="flex justify-between items-center text-xs text-gray-500">
                    <span>{image.settings.dataset}</span>
                    <span>{image.createdAt.toLocaleTimeString()}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* NEW: Auto-Evaluation Suggestion Dialog */}
      {showEvaluationDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg shadow-xl max-w-md w-full p-6 animate-fade-in">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-2">
                <SparklesIcon className="w-6 h-6 text-primary-600" />
                <h3 className="text-lg font-semibold text-gray-900">
                  Try CLIP-based Analysis
                </h3>
              </div>
              <button
                onClick={handleAutoEvaluationDecline}
                className="text-gray-400 hover:text-gray-600 transition-colors"
              >
                <XMarkIcon className="w-5 h-5" />
              </button>
            </div>
            
            <div className="mb-6">
              <p className="text-gray-600 mb-3">
                Great! You've generated {pendingEvaluationData?.images?.length || 0} image(s). 
                Would you like to automatically evaluate how well they match your prompt using our 
                advanced CLIP-based analysis?
              </p>
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                <p className="text-sm text-blue-800">
                  <strong>What you'll get:</strong>
                  <br />• Semantic similarity scores
                  <br />• Detailed keyword analysis  
                  <br />• Feature presence detection
                  <br />• Quality recommendations
                </p>
              </div>
            </div>
            
            <div className="flex space-x-3">
              <button
                onClick={handleAutoEvaluationDecline}
                className="flex-1 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
              >
                No, Thanks
              </button>
              <button
                onClick={handleAutoEvaluationAccept}
                className="flex-1 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
              >
                Yes, Analyze!
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TextToImage;
              
