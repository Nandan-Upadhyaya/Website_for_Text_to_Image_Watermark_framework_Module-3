import React, { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { PhotoIcon, SparklesIcon, PaintBrushIcon, TrashIcon } from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';

const UserGallery = () => {
  const { currentUser, isAuthenticated } = useAuth();
  const [activeTab, setActiveTab] = useState('generated');
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState({
    generated: 0,
    evaluated: 0,
    watermarked: 0,
    total: 0
  });
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  
  const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:5001';

  useEffect(() => {
    if (isAuthenticated) {
      fetchStats();
      fetchImages();
    }
  }, [isAuthenticated, activeTab, page]);

  const fetchStats = async () => {
    try {
      const token = localStorage.getItem('ai_image_suite_auth_token');
      const response = await fetch(`${API_BASE}/api/gallery/stats`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (error) {
      console.error('Error fetching gallery stats:', error);
    }
  };

  const fetchImages = async () => {
    setLoading(true);
    try {
      const token = localStorage.getItem('ai_image_suite_auth_token');
      const endpoint = `${API_BASE}/api/gallery/${activeTab}?page=${page}&per_page=12`;
      
      const response = await fetch(endpoint, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        setImages(data.images);
        setTotalPages(data.total_pages);
      } else {
        toast.error('Failed to load images');
      }
    } catch (error) {
      console.error('Error fetching images:', error);
      toast.error('Error loading gallery');
    } finally {
      setLoading(false);
    }
  };

  const getImageUrl = (imagePath) => {
    if (!imagePath) return '';
    const filename = imagePath.split(/[/\\]/).pop();
    return `${API_BASE}/api/gallery/image/${filename}`;
  };

  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
  };

  const downloadImage = async (imageUrl, filename) => {
    try {
      const response = await fetch(imageUrl);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename || 'image.png';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      toast.success('Image downloaded!');
    } catch (error) {
      console.error('Error downloading image:', error);
      toast.error('Failed to download image');
    }
  };

  const deleteImage = async (imageId) => {
    if (!window.confirm('Are you sure you want to delete this image? This action cannot be undone.')) {
      return;
    }

    try {
      const token = localStorage.getItem('ai_image_suite_auth_token');
      
      if (!token) {
        toast.error('Please sign in to delete images');
        return;
      }
      
      const endpoint = `${API_BASE}/api/gallery/${activeTab}/${imageId}`;
      
      console.log('Deleting image:', { 
        endpoint, 
        imageId, 
        activeTab,
        hasToken: !!token,
        API_BASE 
      });
      
      const response = await fetch(endpoint, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        }
      });

      console.log('Delete response:', {
        status: response.status,
        statusText: response.statusText,
        ok: response.ok
      });

      if (response.ok) {
        toast.success('Image deleted successfully!');
        // Refresh the images and stats
        await fetchImages();
        await fetchStats();
      } else {
        let errorMessage = 'Failed to delete image';
        try {
          const data = await response.json();
          console.error('Delete failed:', data);
          errorMessage = data.error || data.message || errorMessage;
        } catch (e) {
          console.error('Could not parse error response:', e);
        }
        toast.error(errorMessage);
      }
    } catch (error) {
      console.error('Error deleting image:', error);
      toast.error('Network error: ' + error.message);
    }
  };

  const deleteAllImages = async () => {
    const imageCount = stats[activeTab] || 0;
    
    if (imageCount === 0) {
      toast.error('No images to delete');
      return;
    }

    const confirmMessage = `Are you sure you want to delete ALL ${imageCount} ${activeTab} images? This action cannot be undone.`;
    
    if (!window.confirm(confirmMessage)) {
      return;
    }

    try {
      const token = localStorage.getItem('ai_image_suite_auth_token');
      
      if (!token) {
        toast.error('Please sign in to delete images');
        return;
      }
      
      const endpoint = `${API_BASE}/api/gallery/${activeTab}/delete-all`;
      
      console.log('Deleting all images:', { 
        endpoint, 
        activeTab,
        count: imageCount,
        hasToken: !!token,
        API_BASE 
      });
      
      const response = await fetch(endpoint, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        }
      });

      console.log('Delete all response:', {
        status: response.status,
        statusText: response.statusText,
        ok: response.ok
      });

      if (response.ok) {
        const data = await response.json();
        toast.success(`Successfully deleted ${data.count} images!`);
        // Refresh the images and stats
        await fetchImages();
        await fetchStats();
      } else {
        let errorMessage = 'Failed to delete images';
        try {
          const data = await response.json();
          console.error('Delete all failed:', data);
          errorMessage = data.error || data.message || errorMessage;
        } catch (e) {
          console.error('Could not parse error response:', e);
        }
        toast.error(errorMessage);
      }
    } catch (error) {
      console.error('Error deleting all images:', error);
      toast.error('Network error: ' + error.message);
    }
  };

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <PhotoIcon className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-semibold text-gray-900 dark:text-white">
            Sign in required
          </h3>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            Please sign in to view your gallery
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            My Gallery
          </h1>
          <p className="mt-2 text-gray-600 dark:text-gray-400">
            Your personal collection of generated and edited images
          </p>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <div className="flex items-center">
              <SparklesIcon className="h-8 w-8 text-blue-500" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Generated
                </p>
                <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                  {stats.generated}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <div className="flex items-center">
              <PhotoIcon className="h-8 w-8 text-green-500" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Evaluated
                </p>
                <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                  {stats.evaluated}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <div className="flex items-center">
              <PaintBrushIcon className="h-8 w-8 text-purple-500" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Watermarked
                </p>
                <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                  {stats.watermarked}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
            <div className="flex items-center">
              <PhotoIcon className="h-8 w-8 text-indigo-500" />
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600 dark:text-gray-400">
                  Total
                </p>
                <p className="text-2xl font-semibold text-gray-900 dark:text-white">
                  {stats.total}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="border-b border-gray-200 dark:border-gray-700 mb-6">
          <nav className="-mb-px flex space-x-8">
            <button
              onClick={() => { setActiveTab('generated'); setPage(1); }}
              className={`${
                activeTab === 'generated'
                  ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'
              } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
            >
              Generated Images
            </button>
            <button
              onClick={() => { setActiveTab('evaluated'); setPage(1); }}
              className={`${
                activeTab === 'evaluated'
                  ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'
              } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
            >
              Evaluated Images
            </button>
            <button
              onClick={() => { setActiveTab('watermarked'); setPage(1); }}
              className={`${
                activeTab === 'watermarked'
                  ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300 dark:text-gray-400 dark:hover:text-gray-300'
              } whitespace-nowrap py-4 px-1 border-b-2 font-medium text-sm`}
            >
              Watermarked Images
            </button>
          </nav>
        </div>

        {/* Delete All Button */}
        {images.length > 0 && (
          <div className="mb-6 flex justify-end">
            <button
              onClick={deleteAllImages}
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 transition-colors duration-200"
            >
              <TrashIcon className="h-5 w-5 mr-2" />
              Delete All {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)} Images
            </button>
          </div>
        )}

        {/* Images Grid */}
        {loading ? (
          <div className="flex justify-center items-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
          </div>
        ) : images.length === 0 ? (
          <div className="text-center py-12">
            <PhotoIcon className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-semibold text-gray-900 dark:text-white">
              No images yet
            </h3>
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
              Start creating to see your images here!
            </p>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {images.map((image) => (
                <div
                  key={image.id}
                  className="bg-white dark:bg-gray-800 rounded-lg shadow-md overflow-hidden hover:shadow-xl transition-shadow duration-300"
                >
                  <div className="aspect-square relative">
                    <img
                      src={getImageUrl(activeTab === 'generated' ? image.file_path : 
                           activeTab === 'evaluated' ? image.evaluated_image_path : 
                           image.watermarked_image_path)}
                      alt={image.prompt || 'Image'}
                      className="w-full h-full object-cover"
                    />
                  </div>
                  <div className="p-4">
                    {image.prompt && (
                      <p className="text-sm text-gray-700 dark:text-gray-300 mb-2 line-clamp-2">
                        <span className="font-semibold">Prompt:</span> {image.prompt}
                      </p>
                    )}
                    {image.dataset && (
                      <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                        Dataset: {image.dataset}
                      </p>
                    )}
                    {image.watermark_text && (
                      <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                        Watermark: {image.watermark_text}
                      </p>
                    )}
                    {image.watermark_opacity && (
                      <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                        Opacity: {image.watermark_opacity}%
                      </p>
                    )}
                    {image.score !== null && image.score !== undefined && (
                      <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">
                        Score: {image.score.toFixed(2)}
                      </p>
                    )}
                    <p className="text-xs text-gray-400 dark:text-gray-500 mb-3">
                      {formatDate(image.created_at)}
                    </p>
                    <div className="flex gap-2">
                      <button
                        onClick={() => downloadImage(
                          getImageUrl(activeTab === 'generated' ? image.file_path : 
                                    activeTab === 'evaluated' ? image.evaluated_image_path : 
                                    image.watermarked_image_path),
                          `${activeTab}_${image.id}.png`
                        )}
                        className="flex-1 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium py-2 px-4 rounded-md transition-colors duration-200"
                      >
                        Download
                      </button>
                      <button
                        onClick={() => deleteImage(image.id)}
                        className="bg-red-600 hover:bg-red-700 text-white text-sm font-medium py-2 px-4 rounded-md transition-colors duration-200 flex items-center justify-center"
                        title="Delete image"
                      >
                        <TrashIcon className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {/* Pagination */}
            {totalPages > 1 && (
              <div className="mt-8 flex justify-center items-center space-x-4">
                <button
                  onClick={() => setPage(Math.max(1, page - 1))}
                  disabled={page === 1}
                  className="px-4 py-2 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-md disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 dark:hover:bg-gray-700"
                >
                  Previous
                </button>
                <span className="text-gray-700 dark:text-gray-300">
                  Page {page} of {totalPages}
                </span>
                <button
                  onClick={() => setPage(Math.min(totalPages, page + 1))}
                  disabled={page === totalPages}
                  className="px-4 py-2 bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 rounded-md disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50 dark:hover:bg-gray-700"
                >
                  Next
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default UserGallery;
