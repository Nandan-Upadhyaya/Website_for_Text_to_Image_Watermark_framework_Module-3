import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import TextToImage from './pages/TextToImage';
import ImageEvaluator from './pages/ImageEvaluator';
import Watermark from './pages/Watermark';
import Gallery from './pages/Gallery';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
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
            style: {
              background: '#363636',
              color: '#fff',
            },
          }}
        />
      </div>
    </Router>
  );
}

export default App;
