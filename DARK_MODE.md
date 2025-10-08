# Dark/Light Mode Implementation

## Overview
This AI Image Suite now includes a fully functional dark/light mode toggle with smooth animations and comprehensive theme support.

## Features

### 🌓 Theme Toggle Component
- **Location**: `src/components/ThemeToggle.js`
- **Smooth sliding animation** with spring physics
- **Icon transitions** with rotation and fade effects
- **Ripple effect** on click for visual feedback
- **Accessibility support** with proper ARIA labels
- **Hover and active states** with scale animations

### 🎨 Theme Context
- **Location**: `src/contexts/ThemeContext.js`
- **Persistent storage** - remembers user preference
- **System preference detection** - respects OS dark mode setting
- **Global state management** for theme across all components

### 🎯 Enhanced UI Components
- **Navbar**: Full dark mode support with updated colors
- **Dashboard**: Complete theme integration across all sections
- **Cards**: Enhanced shadows and borders for dark mode
- **Forms**: Input fields and textareas with dark mode styling
- **Buttons**: Primary and secondary button variants
- **Scrollbars**: Custom styled for both themes

### ⚡ Smooth Transitions
- **300ms transitions** on all theme-related color changes
- **Spring animations** for the toggle switch (700 stiffness, 35 damping)
- **Framer Motion** powered animations with proper easing curves
- **Staggered animations** for icons and ripple effects

## Usage

### Basic Implementation
The theme toggle is automatically included in the navbar for both desktop and mobile views.

### Using Theme in Components
```javascript
import { useTheme } from '../contexts/ThemeContext';

function MyComponent() {
  const { isDark, toggleTheme } = useTheme();
  
  return (
    <div className={`transition-colors duration-300 ${
      isDark ? 'bg-gray-800 text-white' : 'bg-white text-gray-900'
    }`}>
      {/* Your component content */}
    </div>
  );
}
```

### Tailwind Dark Mode Classes
Use Tailwind's dark mode utilities:
```html
<div className="bg-white dark:bg-gray-800 text-gray-900 dark:text-white transition-colors duration-300">
  Content that adapts to theme
</div>
```

## Technical Details

### Configuration
- **Tailwind**: Configured with `darkMode: 'class'` for class-based toggling
- **CSS Variables**: Used for toast notifications and other dynamic styling
- **Local Storage**: Theme preference saved as 'theme' key

### Animation Specifications
- **Toggle Switch**: Spring animation (stiffness: 700, damping: 35)
- **Icons**: Rotate 360° with scale and opacity transitions
- **Ripple**: Scales from 0.8 to 1.5 with fade out over 600ms
- **Color Transitions**: 300ms ease-in-out for all theme changes

### Accessibility
- Proper ARIA labels for screen readers
- Keyboard navigation support
- Focus rings with theme-appropriate colors
- High contrast maintained in both modes

## Customization

### Adding New Components
When creating new components, ensure they include:
1. Dark mode Tailwind classes (e.g., `dark:bg-gray-800`)
2. Transition classes for smooth color changes
3. Proper contrast ratios for accessibility

### Color Palette
- **Light Mode**: Gray-50/100/200 backgrounds, Gray-900 text
- **Dark Mode**: Gray-800/900 backgrounds, White/Gray-100 text
- **Primary Colors**: Maintained across themes with adjusted opacity
- **Accent Colors**: Purple gradient for branding elements

### Performance
- Minimal re-renders with React Context
- CSS transitions handled by GPU
- Framer Motion animations optimized for performance
- Theme persistence prevents flash of wrong theme

## Browser Support
- Modern browsers with CSS custom properties support
- Fallbacks for older browsers through Tailwind utilities
- Progressive enhancement approach