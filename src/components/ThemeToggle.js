import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { SunIcon, MoonIcon } from '@heroicons/react/24/outline';
import { useTheme } from '../contexts/ThemeContext';

const ThemeToggle = () => {
  const { isDark, toggleTheme } = useTheme();
  const [isAnimating, setIsAnimating] = useState(false);

  const handleToggle = () => {
    if (isAnimating) return; // Prevent multiple rapid clicks
    setIsAnimating(true);
    toggleTheme();
    setTimeout(() => setIsAnimating(false), 600);
  };

  return (
    <div className="flex items-center">
      <motion.button
        onClick={handleToggle}
        disabled={isAnimating}
        className="relative p-1 rounded-full bg-gray-200 dark:bg-gray-700 transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 dark:focus:ring-offset-gray-800 hover:scale-105 active:scale-95"
        aria-label={`Switch to ${isDark ? 'light' : 'dark'} mode`}
        whileTap={{ scale: 0.9 }}
      >
        <div className="relative w-12 h-6 flex items-center">
          {/* Background track with gradient */}
          <motion.div 
            className="absolute inset-0 rounded-full transition-all duration-300"
            animate={{
              background: isDark 
                ? 'linear-gradient(45deg, #1f2937, #374151)' 
                : 'linear-gradient(45deg, #fbbf24, #f59e0b)'
            }}
            style={{
              boxShadow: isDark 
                ? 'inset 0 2px 4px rgba(0, 0, 0, 0.2)' 
                : 'inset 0 2px 4px rgba(0, 0, 0, 0.1)'
            }}
          />
          
          {/* Sliding toggle with enhanced shadow */}
          <motion.div
            className="absolute w-5 h-5 bg-white dark:bg-gray-100 rounded-full flex items-center justify-center"
            animate={{
              x: isDark ? 26 : 2,
              boxShadow: isDark 
                ? '0 2px 8px rgba(0, 0, 0, 0.3)' 
                : '0 2px 6px rgba(0, 0, 0, 0.15)'
            }}
            transition={{
              type: "spring",
              stiffness: 700,
              damping: 35,
            }}
          >
            {/* Icon with enhanced animations */}
            <AnimatePresence mode="wait">
              <motion.div
                key={isDark ? 'moon' : 'sun'}
                initial={{ rotate: -180, scale: 0.5, opacity: 0 }}
                animate={{ rotate: 0, scale: 1, opacity: 1 }}
                exit={{ rotate: 180, scale: 0.5, opacity: 0 }}
                transition={{ 
                  duration: 0.4,
                  ease: [0.4, 0, 0.2, 1]
                }}
                className="w-3 h-3 flex items-center justify-center"
              >
                {isDark ? (
                  <MoonIcon className="w-full h-full text-slate-600" />
                ) : (
                  <SunIcon className="w-full h-full text-yellow-600" />
                )}
              </motion.div>
            </AnimatePresence>
          </motion.div>

          {/* Background icons with improved animations */}
          <div className="absolute inset-0 flex items-center justify-between px-1.5 pointer-events-none overflow-hidden">
            <motion.div
              animate={{ 
                scale: isDark ? 0.8 : 1.2,
                rotate: isDark ? 0 : 360,
                opacity: isDark ? 0.3 : 0
              }}
              transition={{ duration: 0.5 }}
            >
              <SunIcon className="w-3 h-3 text-yellow-300" />
            </motion.div>
            
            <motion.div
              animate={{ 
                scale: isDark ? 1.2 : 0.8,
                rotate: isDark ? 360 : 0,
                opacity: isDark ? 0 : 0.3
              }}
              transition={{ duration: 0.5 }}
            >
              <MoonIcon className="w-3 h-3 text-blue-300" />
            </motion.div>
          </div>

          {/* Ripple effect on click */}
          <AnimatePresence>
            {isAnimating && (
              <motion.div
                className="absolute inset-0 rounded-full bg-primary-400 opacity-30"
                initial={{ scale: 0.8, opacity: 0.6 }}
                animate={{ scale: 1.5, opacity: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.6 }}
              />
            )}
          </AnimatePresence>
        </div>
      </motion.button>
    </div>
  );
};

export default ThemeToggle;