import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HomePage from './pages/HomePage';
import SignInPage from './pages/SignInPage';
import SignUpPage from './pages/SignUpPage';
import UploadPage from './pages/UploadPage';

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/signin" element={<SignInPage/>} />
        <Route path="/signup" element={<SignUpPage />} />
        <Route path="/upload" element={<UploadPage />} />
      </Routes>
    </Router>
  );
};

export default App;
