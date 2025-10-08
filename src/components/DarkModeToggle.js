import React, { useState } from 'react';
import { SunIcon, MoonIcon } from '@heroicons/react/24/outline';
import { useTheme } from '../contexts/ThemeContext';

const DarkModeToggle = () => {
  const { isDark, toggleTheme } = useTheme();
  const [isAnimating, setIsAnimating] = useState(false);

  const handleToggle = () => {
    setIsAnimating(true);
    toggleTheme();
    
    // Reset animation state
    setTimeout(() => {
      setIsAnimating(false);
    }, 600);
  };

  return (
    <button
      onClick={handleToggle}
      className={`
        relative inline-flex h-8 w-16 items-center justify-center rounded-full
        border-2 transition-all duration-300 ease-in-out
        focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2
        ${isDark 
          ? 'bg-slate-800 border-slate-700 focus:ring-offset-slate-900' 
          : 'bg-gray-100 border-gray-200 focus:ring-offset-white'
        }
        hover:scale-105 active:scale-95 group
        ${isAnimating ? 'animate-bounce-gentle' : ''}
      `}
      aria-label={`Switch to ${isDark ? 'light' : 'dark'} mode`}
    >
      {/* Background track with gradient */}
      <div className={`
        absolute inset-1 rounded-full transition-all duration-300
        ${isDark 
          ? 'bg-gradient-to-r from-slate-700 to-slate-600 shadow-inner' 
          : 'bg-gradient-to-r from-white to-gray-50 shadow-sm'
        }
      `} />
      
      {/* Sliding circle with enhanced styling */}
      <div className={`
        absolute top-1 left-1 w-6 h-6 rounded-full
        flex items-center justify-center
        transition-all duration-500 ease-out transform
        ${isDark 
          ? 'translate-x-8 bg-gradient-to-br from-slate-500 to-slate-600 shadow-lg' 
          : 'translate-x-0 bg-gradient-to-br from-white to-gray-100 shadow-md'
        }
        ${isAnimating ? 'scale-110' : 'scale-100'}
        group-hover:shadow-lg
      `}>
        {/* Icons with enhanced animations */}
        <div className="relative w-4 h-4 overflow-hidden">
          <SunIcon className={`
            absolute inset-0 w-4 h-4 text-amber-500 drop-shadow-sm
            transition-all duration-500 ease-out
            ${isDark 
              ? 'opacity-0 scale-0 rotate-180 translate-y-2' 
              : 'opacity-100 scale-100 rotate-0 translate-y-0'
            }
            ${!isDark && isAnimating ? 'animate-pulse' : ''}
          `} />
          <MoonIcon className={`
            absolute inset-0 w-4 h-4 text-indigo-400 drop-shadow-sm
            transition-all duration-500 ease-out
            ${isDark 
              ? 'opacity-100 scale-100 rotate-0 translate-y-0' 
              : 'opacity-0 scale-0 -rotate-180 translate-y-2'
            }
            ${isDark && isAnimating ? 'animate-pulse' : ''}
          `} />
        </div>
        
        {/* Subtle sparkle effect */}
        <div className={`
          absolute inset-0 rounded-full
          ${isAnimating 
            ? isDark 
              ? 'bg-indigo-400/20 animate-ping' 
              : 'bg-amber-400/20 animate-ping'
            : ''
          }
        `} />
      </div>
      
      {/* Enhanced glow effect */}
      <div className={`
        absolute inset-0 rounded-full transition-all duration-300
        ${isDark 
          ? 'shadow-inner shadow-slate-900/50 ring-1 ring-slate-600/50' 
          : 'shadow-inner shadow-gray-900/10 ring-1 ring-gray-200/50'
        }
      `} />
      
      {/* Hover effect overlay */}
      <div className={`
        absolute inset-0 rounded-full opacity-0 group-hover:opacity-100
        transition-opacity duration-300
        ${isDark 
          ? 'bg-slate-600/20' 
          : 'bg-gray-200/30'
        }
      `} />
    </button>
  );
};

export default DarkModeToggle;