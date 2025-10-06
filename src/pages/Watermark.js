import React, { useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { 
  PencilSquareIcon, 
  CloudArrowUpIcon,
  AdjustmentsHorizontalIcon,
  DocumentArrowDownIcon,
  EyeIcon,
  PhotoIcon
} from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';

const Watermark = () => {
  const [images, setImages] = useState([]);
  const [watermarkImage, setWatermarkImage] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processedImages, setProcessedImages] = useState([]);
  const [settings, setSettings] = useState({
    position: 'bottom-right',
    opacity: 80,
    scale: 20,
    paddingX: 5,
    paddingY: 5,
    paddingUnit: 'percentage',
    prefix: '',
    suffix: '',
    autoResize: true
  });

  const [watermarkMode, setWatermarkMode] = useState('image'); // 'image' | 'text'
  const [watermarkFile, setWatermarkFile] = useState(null);
  const [watermarkText, setWatermarkText] = useState('Sample Watermark');
  const [textSize, setTextSize] = useState(32);
  const [textColor, setTextColor] = useState('#FFFFFF');

  const positions = [
    { value: 'top-left', label: 'Top Left' },
    { value: 'top-right', label: 'Top Right' },
    { value: 'bottom-left', label: 'Bottom Left' },
    { value: 'bottom-right', label: 'Bottom Right' },
    { value: 'center', label: 'Center' }
  ];

  const { getRootProps: getImageRootProps, getInputProps: getImageInputProps } = useDropzone({
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.webp']
    },
    onDrop: (acceptedFiles) => {
      const newImages = acceptedFiles.map(file => ({
        id: Date.now() + Math.random(),
        file,
        preview: URL.createObjectURL(file),
        name: file.name
      }));
      setImages(prev => [...prev, ...newImages]);
    },
    multiple: true
  });

  const { getRootProps: getWatermarkRootProps, getInputProps: getWatermarkInputProps } = useDropzone({
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.webp']
    },
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        const file = acceptedFiles[0];
        setWatermarkImage({
          file,
          preview: URL.createObjectURL(file),
          name: file.name
        });
      }
    },
    multiple: false
  });

  const removeImage = (id) => {
    setImages(prev => prev.filter(img => img.id !== id));
  };

  const handleProcess = async () => {
    if (images.length === 0) {
      toast.error('Please select images to watermark');
      return;
    }
    // Require watermark only in image mode
    if (watermarkMode === 'image' && !watermarkImage && !watermarkFile) {
      toast.error('Please select a watermark image');
      return;
    }

    setIsProcessing(true);
    try {
      // Simulate processing time
      await new Promise(resolve => setTimeout(resolve, 2000));

      // Mock processed images
      const processed = images.map(image => ({
        id: image.id,
        originalName: image.name,
        processedName: `${settings.prefix}${image.name.split('.')[0]}${settings.suffix}.${image.name.split('.').pop()}`,
        preview: image.preview, // In real app, this would be the watermarked image
        downloadUrl: image.preview,
        settings: { ...settings }
      }));

      setProcessedImages(processed);
      toast.success(`Successfully watermarked ${images.length} image(s)!`);
    } catch (error) {
      toast.error('Failed to process images. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const downloadAll = () => {
    processedImages.forEach(image => {
      const link = document.createElement('a');
      link.href = image.downloadUrl;
      link.download = image.processedName;
      link.click();
    });
    toast.success('Download started for all images');
  };

  // NEW: Build FormData exactly like standalone settings
  const submitWatermarkJob = async (images) => {
    // images: File[]
    if (!images || images.length === 0) {
      // ...existing code to notify user...
      return;
    }
    if (watermarkMode === 'image' && !watermarkFile) {
      // ...existing code to notify user watermark file missing...
      return;
    }

    const form = new FormData();
    images.forEach((f) => form.append('images', f));
    if (watermarkMode === 'image') {
      form.append('watermark', watermarkFile);
    }
    const settings = {
      pos: settings.position,                                   // 'NW' | 'NE' | 'SW' | 'SE'
      padding: [[settings.paddingX, settings.paddingUnit], [settings.paddingY, settings.paddingUnit]],         // ((x_pad, unit), (y_pad, unit))
      scale: !!settings.autoResize,                         // auto resize
      opacity: Math.max(0, Math.min(1, settings.opacity / 100)), // 0..1 factor
      mode: watermarkMode,                             // 'image' | 'text'
      text: watermarkMode === 'text' ? watermarkText : null,
      text_size: textSize,
      text_color: textColor
    };
    form.append('settings', JSON.stringify(settings));

    // ...existing code to POST to backend...
    // Example:
    /*
    const res = await fetch('/api/watermark', {
      method: 'POST',
      body: form
    });
    if (!res.ok) throw new Error('Watermarking failed');
    // handle response (download zip or array of files)
    */
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Image Watermarking
          </h1>
          <p className="text-gray-600 text-lg max-w-2xl mx-auto">
            Protect your AI-generated images by adding professional watermarks. 
            Upload your images and watermark, then customize the settings.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Input Panel */}
          <div className="lg:col-span-2 space-y-6">
            {/* Image Upload */}
            <div className="card p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center space-x-2">
                <PhotoIcon className="w-5 h-5 text-primary-600" />
                <span>Select Images to Watermark</span>
              </h2>
              
              <div
                {...getImageRootProps()}
                className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-primary-400 transition-colors cursor-pointer"
              >
                <input {...getImageInputProps()} />
                <CloudArrowUpIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                <p className="text-lg text-gray-600 mb-2">
                  Drag & drop images here or click to select
                </p>
                <p className="text-sm text-gray-400">
                  Support for PNG, JPG, JPEG, WebP • Multiple files allowed
                </p>
              </div>

              {images.length > 0 && (
                <div className="mt-4">
                  <h3 className="text-sm font-medium text-gray-700 mb-3">
                    Selected Images ({images.length})
                  </h3>
                  <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-4">
                    {images.map(image => (
                      <div key={image.id} className="relative group">
                        <img
                          src={image.preview}
                          alt={image.name}
                          className="w-full aspect-square object-cover rounded-lg"
                        />
                        <button
                          onClick={() => removeImage(image.id)}
                          className="absolute -top-2 -right-2 w-6 h-6 bg-red-500 text-white rounded-full text-xs hover:bg-red-600 opacity-0 group-hover:opacity-100 transition-opacity"
                        >
                          ×
                        </button>
                        <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-50 text-white text-xs p-2 rounded-b-lg opacity-0 group-hover:opacity-100 transition-opacity">
                          {image.name}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Watermark Upload */}
            <div className="card p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center space-x-2">
                <PencilSquareIcon className="w-5 h-5 text-primary-600" />
                <span>Select Watermark Image</span>
              </h2>
              
              <div
                {...getWatermarkRootProps()}
                className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-primary-400 transition-colors cursor-pointer"
              >
                <input {...getWatermarkInputProps()} />
                {watermarkImage ? (
                  <div className="space-y-4">
                    <img 
                      src={watermarkImage.preview} 
                      alt="Watermark" 
                      className="max-h-32 mx-auto rounded-lg"
                    />
                    <p className="text-sm text-gray-600">
                      {watermarkImage.name} • Click to replace
                    </p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <PencilSquareIcon className="w-8 h-8 text-gray-400 mx-auto" />
                    <div>
                      <p className="text-gray-600">Select watermark image</p>
                      <p className="text-sm text-gray-400">PNG recommended for transparency</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Settings Panel */}
          <div className="space-y-6">
            <div className="card p-6">
              <div className="flex items-center space-x-2 mb-4">
                <AdjustmentsHorizontalIcon className="w-5 h-5 text-primary-600" />
                <h2 className="text-xl font-semibold">Watermark Settings</h2>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Mode
                  </label>
                  <div className="flex space-x-4">
                    <label className="inline-flex items-center">
                      <input type="radio" name="wm-mode" value="image" checked={watermarkMode==='image'} onChange={() => setWatermarkMode('image')} />
                      <span className="ml-2">Image</span>
                    </label>
                    <label className="inline-flex items-center">
                      <input type="radio" name="wm-mode" value="text" checked={watermarkMode==='text'} onChange={() => setWatermarkMode('text')} />
                      <span className="ml-2">Text</span>
                    </label>
                  </div>
                </div>

                {watermarkMode === 'image' ? (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Watermark source
                    </label>
                    <input type="file" accept="image/*" onChange={(e)=> setWatermarkFile(e.target.files?.[0] || null)} />
                  </div>
                ) : (
                  <>
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-2">
                        Text watermark
                      </label>
                      <input className="input w-full" value={watermarkText} onChange={(e)=> setWatermarkText(e.target.value)} placeholder="Enter watermark text" />
                    </div>
                    <div className="flex space-x-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Text size
                        </label>
                        <input type="number" min="6" className="input w-24" value={textSize} onChange={(e)=> setTextSize(parseInt(e.target.value || 0, 10))} />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-2">
                          Color (name or hex)
                        </label>
                        <input className="input w-40" value={textColor} onChange={(e)=> setTextColor(e.target.value)} placeholder="#FFFFFF or red" />
                      </div>
                    </div>
                  </>
                )}

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Position
                  </label>
                  <div className="grid grid-cols-2 gap-2">
                    <label className="inline-flex items-center"><input type="radio" name="pos" value="NW" checked={settings.position==='NW'} onChange={()=> setSettings({...settings, position:'NW'})} /><span className="ml-2">Top left</span></label>
                    <label className="inline-flex items-center"><input type="radio" name="pos" value="NE" checked={settings.position==='NE'} onChange={()=> setSettings({...settings, position:'NE'})} /><span className="ml-2">Top right</span></label>
                    <label className="inline-flex items-center"><input type="radio" name="pos" value="SW" checked={settings.position==='SW'} onChange={()=> setSettings({...settings, position:'SW'})} /><span className="ml-2">Bottom left</span></label>
                    <label className="inline-flex items-center"><input type="radio" name="pos" value="SE" checked={settings.position==='SE'} onChange={()=> setSettings({...settings, position:'SE'})} /><span className="ml-2">Bottom right</span></label>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Padding X: {settings.paddingX}
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="20"
                      value={settings.paddingX}
                      onChange={(e) => setSettings({...settings, paddingX: parseInt(e.target.value)})}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Padding Y: {settings.paddingY}
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="20"
                      value={settings.paddingY}
                      onChange={(e) => setSettings({...settings, paddingY: parseInt(e.target.value)})}
                      className="w-full"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    File Prefix
                  </label>
                  <input
                    type="text"
                    value={settings.prefix}
                    onChange={(e) => setSettings({...settings, prefix: e.target.value})}
                    className="input"
                    placeholder="watermarked_"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    File Suffix
                  </label>
                  <input
                    type="text"
                    value={settings.suffix}
                    onChange={(e) => setSettings({...settings, suffix: e.target.value})}
                    className="input"
                    placeholder="_protected"
                  />
                </div>

                <div className="flex items-center">
                  <input
                    type="checkbox"
                    id="autoResize"
                    checked={settings.autoResize}
                    onChange={(e) => setSettings({...settings, autoResize: e.target.checked})}
                    className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                  />
                  <label htmlFor="autoResize" className="ml-2 text-sm text-gray-700">
                    Auto-resize watermark
                  </label>
                </div>
              </div>

              <button
                onClick={handleProcess}
                // Enable button in text mode even without a watermark image/file
                disabled={
                  isProcessing ||
                  images.length === 0 ||
                  (watermarkMode === 'image' && !watermarkImage && !watermarkFile)
                }
                className="btn-primary w-full mt-6 flex items-center justify-center space-x-2"
              >
                {isProcessing ? (
                  <>
                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span>Processing...</span>
                  </>
                ) : (
                  <>
                    <PencilSquareIcon className="w-5 h-5" />
                    <span>Apply Watermarks</span>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Processed Images */}
        {processedImages.length > 0 && (
          <div className="mt-8">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-2xl font-bold text-gray-900">
                Processed Images ({processedImages.length})
              </h2>
              <button
                onClick={downloadAll}
                className="btn-primary flex items-center space-x-2"
              >
                <DocumentArrowDownIcon className="w-5 h-5" />
                <span>Download All</span>
              </button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {processedImages.map((image) => (
                <div key={image.id} className="card p-4">
                  <div className="relative group mb-3">
                    <img
                      src={image.preview}
                      alt={image.processedName}
                      className="w-full aspect-square object-cover rounded-lg"
                    />
                    <div className="absolute inset-0 bg-black bg-opacity-50 opacity-0 group-hover:opacity-100 transition-opacity rounded-lg flex items-center justify-center">
                      <button className="p-2 bg-white bg-opacity-20 rounded-full text-white hover:bg-opacity-30 transition-colors">
                        <EyeIcon className="w-6 h-6" />
                      </button>
                    </div>
                  </div>
                  
                  <div className="space-y-2">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      {image.processedName}
                    </p>
                    <p className="text-xs text-gray-500">
                      Original: {image.originalName}
                    </p>
                    <button
                      onClick={() => {
                        const link = document.createElement('a');
                        link.href = image.downloadUrl;
                        link.download = image.processedName;
                        link.click();
                      }}
                      className="btn-secondary w-full text-sm flex items-center justify-center space-x-1"
                    >
                      <DocumentArrowDownIcon className="w-4 h-4" />
                      <span>Download</span>
                    </button>
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

export default Watermark;
