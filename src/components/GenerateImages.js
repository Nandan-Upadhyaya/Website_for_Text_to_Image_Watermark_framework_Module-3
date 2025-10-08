import React, { useState } from 'react';
import { saveGeneratedImageToGallery } from '../utils/helpers';

const GenerateImages = () => {
  const [prompt, setPrompt] = useState('');
  const [seed, setSeed] = useState('');
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [generationSteps, setGenerationSteps] = useState(50);
  const [generatedImage, setGeneratedImage] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);

  const handleImageGeneration = async () => {
    try {
      setIsGenerating(true);
      
      // Replace this section with your actual API call or image generation logic
      const response = await fetch('your-api-endpoint', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          seed: seed || Math.floor(Math.random() * 2147483647), // Use provided seed or generate random one
          guidance_scale: guidanceScale,
          num_inference_steps: generationSteps
        })
      });
      
      if (!response.ok) throw new Error('Image generation failed');
      
      // Get the image data
      const imageBlob = await response.blob();
      
      // Save the current seed value (in case it was randomly generated)
      const currentSeed = seed || Math.floor(Math.random() * 2147483647);
      setSeed(currentSeed);
      
      // Update UI with generated image
      setGeneratedImage(imageBlob);
      
      // Save to gallery
      await saveGeneratedImageToGallery({
        imageFile: imageBlob,
        prompt,
        seed: currentSeed,
        guidanceScale,
        generationSteps
      });
      
      console.log('Image saved to gallery successfully');
    } catch (error) {
      console.error('Error generating image:', error);
      alert('Failed to generate image. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="container mx-auto p-4">
      <div className="mb-4">
        <label className="block text-sm font-medium mb-1">Prompt</label>
        <textarea
          className="w-full p-2 border rounded"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Describe the image you want to generate..."
          rows={3}
        />
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
        <div>
          <label className="block text-sm font-medium mb-1">Seed (optional)</label>
          <input
            type="number"
            className="w-full p-2 border rounded"
            value={seed}
            onChange={(e) => setSeed(e.target.value)}
            placeholder="Leave empty for random seed"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-1">Guidance Scale ({guidanceScale})</label>
          <input
            type="range"
            min="1"
            max="20"
            step="0.1"
            className="w-full"
            value={guidanceScale}
            onChange={(e) => setGuidanceScale(parseFloat(e.target.value))}
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-1">Steps ({generationSteps})</label>
          <input
            type="range"
            min="10"
            max="150"
            step="1"
            className="w-full"
            value={generationSteps}
            onChange={(e) => setGenerationSteps(parseInt(e.target.value))}
          />
        </div>
      </div>
      
      <button
        className="bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 disabled:bg-blue-300"
        onClick={handleImageGeneration}
        disabled={isGenerating || !prompt.trim()}
      >
        {isGenerating ? 'Generating...' : 'Generate Image'}
      </button>
      
      {generatedImage && (
        <div className="mt-6">
          <h3 className="text-lg font-medium mb-2">Generated Image</h3>
          <div className="border rounded p-2">
            <img 
              src={URL.createObjectURL(generatedImage)} 
              alt="Generated" 
              className="max-w-full h-auto"
            />
            <div className="mt-2 text-sm">
              <p><strong>Prompt:</strong> {prompt}</p>
              <p><strong>Seed:</strong> {seed}</p>
              <p><strong>Guidance Scale:</strong> {guidanceScale}</p>
              <p><strong>Steps:</strong> {generationSteps}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default GenerateImages;