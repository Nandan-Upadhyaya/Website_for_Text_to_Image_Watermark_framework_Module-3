import React, { useState } from 'react';
import { PhotoIcon, SparklesIcon, AdjustmentsHorizontalIcon } from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';

const TextToImage = () => {
  const [prompt, setPrompt] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImages, setGeneratedImages] = useState([]);
  const [settings, setSettings] = useState({
    model: 'df-gan',
    dataset: 'bird',
    batchSize: 1,
    steps: 100,
    guidance: 7.5,
    seed: -1
  });

  const datasets = [
    { value: 'bird', label: 'Birds (CUB-200-2011)' },
    { value: 'coco', label: 'COCO Dataset' },
    { value: 'flower', label: 'Oxford Flowers' }
  ];

  const examplePrompts = {
    bird: [
      "A small yellow bird with black wings perched on a branch",
      "A red cardinal sitting on a snowy pine tree",
      "A blue jay with distinctive crest feathers",
      "A tiny hummingbird hovering near purple flowers"
    ],
    coco: [
      "A cat sitting on a wooden table in a sunny kitchen",
      "A dog playing fetch in a green park",
      "A bicycle leaning against a brick wall",
      "Children playing soccer in a field"
    ],
    flower: [
      "A single red rose with morning dew drops",
      "A field of sunflowers facing the sun",
      "Purple lavender swaying in the breeze",
      "A bouquet of colorful tulips in spring"
    ]
  };

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      toast.error('Please enter a text prompt');
      return;
    }

    setIsGenerating(true);
    try {
      // Simulate API call to DF-GAN backend
      await new Promise(resolve => setTimeout(resolve, 3000));
      
      // Mock generated images
      const mockImages = Array.from({ length: settings.batchSize }, (_, i) => ({
        id: Date.now() + i,
        url: `https://picsum.photos/512/512?random=${Date.now() + i}`,
        prompt: prompt,
        settings: { ...settings },
        createdAt: new Date()
      }));

      setGeneratedImages(prev => [...mockImages, ...prev]);
      toast.success(`Generated ${settings.batchSize} image(s) successfully!`);
    } catch (error) {
      toast.error('Failed to generate images. Please try again.');
    } finally {
      setIsGenerating(false);
    }
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
                    Batch Size: {settings.batchSize}
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="4"
                    value={settings.batchSize}
                    onChange={(e) => setSettings({...settings, batchSize: parseInt(e.target.value)})}
                    className="w-full"
                    disabled={isGenerating}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Generation Steps: {settings.steps}
                  </label>
                  <input
                    type="range"
                    min="20"
                    max="200"
                    step="10"
                    value={settings.steps}
                    onChange={(e) => setSettings({...settings, steps: parseInt(e.target.value)})}
                    className="w-full"
                    disabled={isGenerating}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Guidance Scale: {settings.guidance}
                  </label>
                  <input
                    type="range"
                    min="1"
                    max="20"
                    step="0.5"
                    value={settings.guidance}
                    onChange={(e) => setSettings({...settings, guidance: parseFloat(e.target.value)})}
                    className="w-full"
                    disabled={isGenerating}
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Seed (-1 for random)
                  </label>
                  <input
                    type="number"
                    value={settings.seed}
                    onChange={(e) => setSettings({...settings, seed: parseInt(e.target.value)})}
                    className="input"
                    disabled={isGenerating}
                    placeholder="-1"
                  />
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Generated Images */}
        {generatedImages.length > 0 && (
          <div className="mt-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">Generated Images</h2>
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
    </div>
  );
};

export default TextToImage;
