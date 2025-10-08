import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { ThemeProvider } from './contexts/ThemeContext';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import TextToImage from './pages/TextToImage';
import ImageEvaluator from './pages/ImageEvaluator';
import Watermark from './pages/Watermark';
import Gallery from './pages/Gallery';

function App() {
  return (
    <ThemeProvider>
      <Router>
        <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-300">
          <Navbar />
          <main className="pt-16">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/generate" element={<TextToImage />} />
              <Route path="/evaluate" element={<ImageEvaluator />} />
              <Route path="/watermark" element={<Watermark />} />
              <Route path="/gallery" element={<Gallery />} />
            </Routes>
          </main>
          <Toaster 
            position="top-right"
            toastOptions={{
              duration: 4000,
              className: 'dark:!bg-gray-800 dark:!text-white dark:!border-gray-700',
              style: {
                background: '#363636',
                color: '#fff',
                borderRadius: '8px',
                boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
              },
            }}
          />
        </div>
      </Router>
    </ThemeProvider>
  );
}

export default App;
