import React from 'react';
import { Link } from 'react-router-dom';
import { 
  PhotoIcon, 
  CheckCircleIcon, 
  PencilSquareIcon,
  SparklesIcon,
  ChartBarIcon,
  ShieldCheckIcon
} from '@heroicons/react/24/outline';

const Dashboard = () => {
  const features = [
    {
      title: 'Text-to-Image Generation',
      description: 'Create stunning images from text descriptions using the advanced DF-GAN model.',
      icon: PhotoIcon,
      href: '/generate',
      color: 'from-blue-500 to-blue-600',
      stats: 'AI-Powered'
    },
    {
      title: 'Image Quality Evaluation',
      description: 'Analyze how well generated images match their text prompts using CLIP-based evaluation.',
      icon: CheckCircleIcon,
      href: '/evaluate',
      color: 'from-green-500 to-green-600',
      stats: 'CLIP Model'
    },
    {
      title: 'Watermark Protection',
      description: 'Add professional watermarks to protect your generated images with customizable options.',
      icon: PencilSquareIcon,
      href: '/watermark',
      color: 'from-purple-500 to-purple-600',
      stats: 'Bulk Process'
    }
  ];

  const stats = [
    { label: 'Images Generated', value: '1,234', icon: SparklesIcon },
    { label: 'Evaluations Run', value: '567', icon: ChartBarIcon },
    { label: 'Images Protected', value: '890', icon: ShieldCheckIcon },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50">
      {/* Hero Section */}
      <div className="container py-16">
        <div className="text-center max-w-4xl mx-auto">
          <h1 className="text-5xl font-bold text-gray-900 mb-6">
            AI Image Processing
            <span className="block text-transparent bg-clip-text bg-gradient-to-r from-primary-600 to-accent-600">
              Suite
            </span>
          </h1>
          <p className="text-xl text-gray-600 mb-8 leading-relaxed">
            Generate, evaluate, and protect your AI-created images with our comprehensive suite of tools.
            Powered by cutting-edge machine learning models for professional results.
          </p>
          <Link
            to="/generate"
            className="btn-primary text-lg px-8 py-3 inline-flex items-center space-x-2"
          >
            <SparklesIcon className="w-5 h-5" />
            <span>Start Creating</span>
          </Link>
        </div>
      </div>

      {/* Stats Section */}
      <div className="container pb-16">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-16">
          {stats.map((stat, index) => {
            const IconComponent = stat.icon;
            return (
              <div key={index} className="card p-6 text-center">
                <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                  <IconComponent className="w-6 h-6 text-primary-600" />
                </div>
                <div className="text-3xl font-bold text-gray-900 mb-2">{stat.value}</div>
                <div className="text-gray-600">{stat.label}</div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Features Grid */}
      <div className="container pb-16">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            Three Powerful Modules
          </h2>
          <p className="text-gray-600 text-lg max-w-2xl mx-auto">
            Each module is designed to handle a specific aspect of AI image processing,
            working together seamlessly for your workflow.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => {
            const IconComponent = feature.icon;
            return (
              <Link
                key={index}
                to={feature.href}
                className="group card p-8 hover:shadow-lg transition-all duration-300 hover:-translate-y-1"
              >
                <div className={`w-16 h-16 bg-gradient-to-r ${feature.color} rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-transform duration-300`}>
                  <IconComponent className="w-8 h-8 text-white" />
                </div>
                
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-xl font-semibold text-gray-900 group-hover:text-primary-600 transition-colors">
                    {feature.title}
                  </h3>
                  <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded-full">
                    {feature.stats}
                  </span>
                </div>
                
                <p className="text-gray-600 leading-relaxed mb-6">
                  {feature.description}
                </p>
                
                <div className="flex items-center text-primary-600 font-medium group-hover:text-primary-700">
                  <span>Get Started</span>
                  <svg className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </div>
              </Link>
            );
          })}
        </div>
      </div>

      {/* Workflow Section */}
      <div className="bg-white py-16">
        <div className="container">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">
              Simple Three-Step Workflow
            </h2>
            <p className="text-gray-600 text-lg">
              From concept to protected final product in just three steps
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-blue-600">1</span>
              </div>
              <h3 className="text-xl font-semibold mb-2">Generate</h3>
              <p className="text-gray-600">Create images from text descriptions using DF-GAN</p>
            </div>
            
            <div className="text-center">
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-green-600">2</span>
              </div>
              <h3 className="text-xl font-semibold mb-2">Evaluate</h3>
              <p className="text-gray-600">Assess quality and prompt matching accuracy</p>
            </div>
            
            <div className="text-center">
              <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-2xl font-bold text-purple-600">3</span>
              </div>
              <h3 className="text-xl font-semibold mb-2">Protect</h3>
              <p className="text-gray-600">Add watermarks to secure your creations</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
