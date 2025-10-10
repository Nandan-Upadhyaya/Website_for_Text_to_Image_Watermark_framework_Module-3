import React from 'react';
import { Link } from 'react-router-dom';
import { 
  PhotoIcon,
  HeartIcon,
  MapPinIcon,
  EnvelopeIcon,
  PhoneIcon
} from '@heroicons/react/24/outline';

const Footer = () => {
  const currentYear = new Date().getFullYear();

  const navigation = {
    main: [
      { name: 'Dashboard', href: '/' },
      { name: 'Generate Images', href: '/generate' },
      { name: 'Evaluate Images', href: '/evaluate' },
      { name: 'Add Watermark', href: '/watermark' },
      { name: 'Gallery', href: '/gallery' },
    ],
  };

  return (
    <footer className="bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700 transition-colors duration-300">
      <div className="container">
        {/* Main Footer Content */}
        <div className="py-12 lg:py-16">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 lg:gap-12">
            {/* Brand Section */}
            <div className="footer-section">
              <Link to="/" className="flex items-center space-x-2 mb-4">
                <div className="w-8 h-8 bg-gradient-to-br from-primary-500 to-accent-500 rounded-lg flex items-center justify-center">
                  <PhotoIcon className="w-5 h-5 text-white" />
                </div>
                <span className="text-xl font-bold bg-gradient-to-r from-primary-600 to-accent-600 bg-clip-text text-transparent">
                  AI Image Suite
                </span>
              </Link>
              
              {/* Contact Info */}
              <div className="space-y-2">
                <div className="flex items-center space-x-3 text-sm text-gray-600 dark:text-gray-400">
                  <EnvelopeIcon className="w-4 h-4 text-primary-500" />
                  <span>contact@aiimagesuite.com</span>
                </div>
                <div className="flex items-center space-x-3 text-sm text-gray-600 dark:text-gray-400">
                  <PhoneIcon className="w-4 h-4 text-primary-500" />
                  <span>+1 (555) 123-4567</span>
                </div>
                <div className="flex items-center space-x-3 text-sm text-gray-600 dark:text-gray-400">
                  <MapPinIcon className="w-4 h-4 text-primary-500" />
                  <span>San Francisco, CA</span>
                </div>
              </div>
            </div>

            {/* Quick Links */}
            <div className="footer-section">
              <h3 className="text-sm font-semibold text-gray-900 dark:text-white uppercase tracking-wider mb-4">
                Quick Links
              </h3>
              <div className="flex flex-wrap gap-x-6 gap-y-2">
                {navigation.main.map((item) => (
                  <Link
                    key={item.name}
                    to={item.href}
                    className="text-sm text-gray-600 dark:text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 transition-colors duration-200"
                  >
                    {item.name}
                  </Link>
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="border-t border-gray-200 dark:border-gray-700 py-6">
          <div className="flex flex-col md:flex-row justify-between items-center space-y-4 md:space-y-0">
            <div className="flex items-center space-x-1 text-sm text-gray-600 dark:text-gray-400">
              <span>© {currentYear} AI Image Suite. Made with</span>
              <HeartIcon className="w-4 h-4 text-red-500" />
              <span>by The GANners Team</span>
            </div>
            
            <div className="flex items-center space-x-6 text-sm text-gray-600 dark:text-gray-400">
              <span>Powered by AI</span>
              <div className="w-1 h-1 bg-gray-400 rounded-full"></div>
              <span>Built with React & Tailwind</span>
            </div>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;