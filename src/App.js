import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import { ThemeProvider } from './contexts/ThemeContext';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import LoginModal from './components/LoginModal';
import Dashboard from './pages/Dashboard';
import TextToImage from './pages/TextToImage';
import ImageEvaluator from './pages/ImageEvaluator';
import Watermark from './pages/Watermark';
import UserGallery from './pages/UserGallery';
import Login from './pages/Login';
import Signup from './pages/Signup';

function AppContent() {
  const { showLoginModal, closeLoginModal } = useAuth();

  return (
    <>
      <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-300 flex flex-col">
        <Navbar />
        <main className="pt-16 flex-grow">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/generate" element={<TextToImage />} />
            <Route path="/evaluate" element={<ImageEvaluator />} />
            <Route path="/watermark" element={<Watermark />} />
            <Route path="/gallery" element={<UserGallery />} />
            <Route path="/login" element={<Login />} />
            <Route path="/signup" element={<Signup />} />
          </Routes>
        </main>
        <Footer />
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
      <LoginModal isOpen={showLoginModal} onClose={closeLoginModal} />
    </>
  );
}

function App() {
  return (
    <ThemeProvider>
      <Router>
        <AuthProvider>
          <AppContent />
        </AuthProvider>
      </Router>
    </ThemeProvider>
  );
}

export default App;
