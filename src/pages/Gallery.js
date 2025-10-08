import React, { useState } from 'react';
import { 
  RectangleGroupIcon,
  MagnifyingGlassIcon,
  FunnelIcon,
  EyeIcon,
  HeartIcon,
  ShareIcon,
  DocumentArrowDownIcon,
  TagIcon
} from '@heroicons/react/24/outline';

const Gallery = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterCategory, setFilterCategory] = useState('all');
  const [sortBy, setSortBy] = useState('newest');

  // Mock gallery data
  const [images] = useState([
    {
      id: 1,
      url: 'https://picsum.photos/400/400?random=1',
      prompt: 'A small yellow bird with black wings perched on a branch',
      category: 'generated',
      score: 85,
      createdAt: '2024-03-15T10:30:00Z',
      liked: false,
      tags: ['bird', 'yellow', 'nature']
    },
    {
      id: 2,
      url: 'https://picsum.photos/400/400?random=2',
      prompt: 'A red cardinal sitting on a snowy pine tree',
      category: 'generated',
      score: 92,
      createdAt: '2024-03-15T09:15:00Z',
      liked: true,
      tags: ['bird', 'red', 'winter', 'snow']
    },
    {
      id: 3,
      url: 'https://picsum.photos/400/400?random=3',
      prompt: 'A cat sitting on a wooden table in a sunny kitchen',
      category: 'generated',
      score: 78,
      createdAt: '2024-03-15T08:45:00Z',
      liked: false,
      tags: ['cat', 'kitchen', 'sunny']
    },
    {
      id: 4,
      url: 'https://picsum.photos/400/400?random=4',
      prompt: 'Original uploaded image for watermarking',
      category: 'watermarked',
      score: null,
      createdAt: '2024-03-15T07:20:00Z',
      liked: true,
      tags: ['watermarked', 'protected']
    },
    {
      id: 5,
      url: 'https://picsum.photos/400/400?random=5',
      prompt: 'A field of sunflowers facing the sun',
      category: 'generated',
      score: 88,
      createdAt: '2024-03-14T16:30:00Z',
      liked: false,
      tags: ['flowers', 'sunflowers', 'field', 'sun']
    },
    {
      id: 6,
      url: 'https://picsum.photos/400/400?random=6',
      prompt: 'Evaluated image from external source',
      category: 'evaluated',
      score: 73,
      createdAt: '2024-03-14T15:10:00Z',
      liked: false,
      tags: ['evaluated', 'analysis']
    }
  ]);

  const categories = [
    { value: 'all', label: 'All Images', count: images.length },
    { value: 'generated', label: 'Generated', count: images.filter(img => img.category === 'generated').length },
    { value: 'evaluated', label: 'Evaluated', count: images.filter(img => img.category === 'evaluated').length },
    { value: 'watermarked', label: 'Watermarked', count: images.filter(img => img.category === 'watermarked').length }
  ];

  const sortOptions = [
    { value: 'newest', label: 'Newest First' },
    { value: 'oldest', label: 'Oldest First' },
    { value: 'highest-score', label: 'Highest Score' },
    { value: 'lowest-score', label: 'Lowest Score' }
  ];

  const filteredImages = images
    .filter(image => {
      const matchesSearch = image.prompt.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           image.tags.some(tag => tag.toLowerCase().includes(searchTerm.toLowerCase()));
      const matchesCategory = filterCategory === 'all' || image.category === filterCategory;
      return matchesSearch && matchesCategory;
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'newest':
          return new Date(b.createdAt) - new Date(a.createdAt);
        case 'oldest':
          return new Date(a.createdAt) - new Date(b.createdAt);
        case 'highest-score':
          return (b.score || 0) - (a.score || 0);
        case 'lowest-score':
          return (a.score || 0) - (b.score || 0);
        default:
          return 0;
      }
    });

  const getCategoryColor = (category) => {
    switch (category) {
      case 'generated': return 'bg-blue-100 text-blue-800';
      case 'evaluated': return 'bg-green-100 text-green-800';
      case 'watermarked': return 'bg-purple-100 text-purple-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getScoreColor = (score) => {
    if (!score) return 'text-gray-400';
    if (score >= 80) return 'text-green-600';
    if (score >= 60) return 'text-blue-600';
    if (score >= 40) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 py-8">
      <div className="container">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-gray-100 mb-4">
            Image Gallery
          </h1>
          <p className="text-gray-600 text-lg max-w-2xl mx-auto">
            Browse and manage all your AI-generated, evaluated, and watermarked images in one place.
          </p>
        </div>

        {/* Filters and Search */}
        <div className="card p-6 mb-8">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between space-y-4 lg:space-y-0 lg:space-x-6">
            {/* Search */}
            <div className="flex-1 relative">
              <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                placeholder="Search images by prompt or tags..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="input pl-10 w-full"
              />
            </div>

            {/* Category Filter */}
            <div className="flex items-center space-x-2">
              <FunnelIcon className="w-5 h-5 text-gray-400" />
              <select
                value={filterCategory}
                onChange={(e) => setFilterCategory(e.target.value)}
                className="input min-w-0"
              >
                {categories.map(category => (
                  <option key={category.value} value={category.value}>
                    {category.label} ({category.count})
                  </option>
                ))}
              </select>
            </div>

            {/* Sort */}
            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-600 whitespace-nowrap">Sort by:</span>
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                className="input min-w-0"
              >
                {sortOptions.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* Results Count */}
        <div className="flex justify-between items-center mb-6">
          <p className="text-gray-600">
            {filteredImages.length} image{filteredImages.length !== 1 ? 's' : ''} found
          </p>
        </div>

        {/* Gallery Grid */}
        {filteredImages.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {filteredImages.map((image) => (
              <div key={image.id} className="card p-0 overflow-hidden group hover:shadow-lg transition-all duration-300">
                {/* Image */}
                <div className="relative aspect-square">
                  <img
                    src={image.url}
                    alt={image.prompt}
                    className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-105"
                  />
                  
                  {/* Overlay */}
                  <div className="absolute inset-0 bg-black bg-opacity-0 group-hover:bg-opacity-40 transition-all duration-300 flex items-center justify-center">
                    <div className="opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex space-x-2">
                      <button className="p-2 bg-white bg-opacity-20 rounded-full text-white hover:bg-opacity-30 transition-colors">
                        <EyeIcon className="w-5 h-5" />
                      </button>
                      <button className="p-2 bg-white bg-opacity-20 rounded-full text-white hover:bg-opacity-30 transition-colors">
                        <ShareIcon className="w-5 h-5" />
                      </button>
                      <button className="p-2 bg-white bg-opacity-20 rounded-full text-white hover:bg-opacity-30 transition-colors">
                        <DocumentArrowDownIcon className="w-5 h-5" />
                      </button>
                    </div>
                  </div>

                  {/* Category Badge */}
                  <div className="absolute top-3 left-3">
                    <span className={`px-2 py-1 text-xs font-medium rounded-full ${getCategoryColor(image.category)}`}>
                      {image.category}
                    </span>
                  </div>

                  {/* Score Badge */}
                  {image.score && (
                    <div className="absolute top-3 right-3">
                      <span className={`px-2 py-1 text-xs font-bold bg-white rounded-full ${getScoreColor(image.score)}`}>
                        {image.score}%
                      </span>
                    </div>
                  )}
                </div>

                {/* Content */}
                <div className="p-4">
                  <p className="text-sm text-gray-600 line-clamp-2 mb-3">
                    {image.prompt}
                  </p>

                  {/* Tags */}
                  {image.tags.length > 0 && (
                    <div className="flex flex-wrap gap-1 mb-3">
                      {image.tags.slice(0, 3).map((tag, index) => (
                        <span key={index} className="inline-flex items-center px-2 py-1 text-xs bg-gray-100 text-gray-600 rounded">
                          <TagIcon className="w-3 h-3 mr-1" />
                          {tag}
                        </span>
                      ))}
                      {image.tags.length > 3 && (
                        <span className="text-xs text-gray-400">
                          +{image.tags.length - 3} more
                        </span>
                      )}
                    </div>
                  )}

                  {/* Footer */}
                  <div className="flex justify-between items-center text-xs text-gray-500">
                    <span>
                      {new Date(image.createdAt).toLocaleDateString()}
                    </span>
                    <button 
                      className={`p-1 rounded transition-colors ${
                        image.liked 
                          ? 'text-red-500 hover:text-red-600' 
                          : 'text-gray-400 hover:text-red-500'
                      }`}
                    >
                      <HeartIcon className={`w-4 h-4 ${image.liked ? 'fill-current' : ''}`} />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-12">
            <RectangleGroupIcon className="w-16 h-16 text-gray-300 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No images found</h3>
            <p className="text-gray-500">
              {searchTerm || filterCategory !== 'all' 
                ? 'Try adjusting your search or filter criteria'
                : 'Start generating, evaluating, or watermarking images to see them here'
              }
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Gallery;
