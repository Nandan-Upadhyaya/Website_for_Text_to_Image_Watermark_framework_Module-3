import React, { useEffect, useState } from 'react';
import { loadGalleryImages, formatRelativeTime } from '../utils/helpers';

const Gallery = () => {
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Load saved images from localStorage
    const savedImages = loadGalleryImages();
    setImages(savedImages);
    setLoading(false);
  }, []);

  if (loading) {
    return <div className="text-center py-10">Loading gallery...</div>;
  }

  if (images.length === 0) {
    return (
      <div className="text-center py-10">
        <p className="text-gray-600">No generated images yet. Generate some images to see them here.</p>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <h2 className="text-2xl font-bold mb-6">Your Generated Images</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {images.map((item) => (
          <div key={item.id} className="border rounded-lg overflow-hidden shadow-lg">
            <img 
              src={item.image} 
              alt={item.prompt} 
              className="w-full h-64 object-cover"
            />
            <div className="p-4">
              <p className="font-semibold mb-2 text-sm line-clamp-2">{item.prompt}</p>
              <div className="grid grid-cols-2 gap-2 text-xs text-gray-600">
                <div>
                  <span className="font-medium">Seed:</span> {item.seed}
                </div>
                <div>
                  <span className="font-medium">Guidance:</span> {item.guidanceScale}
                </div>
                <div>
                  <span className="font-medium">Steps:</span> {item.generationSteps}
                </div>
                <div>
                  <span className="font-medium">Created:</span> {formatRelativeTime(item.createdAt)}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Gallery;
