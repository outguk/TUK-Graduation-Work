import { ThemeProvider, CssBaseline } from '@mui/material';
import theme from './styles/theme';
import { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LoadingSpinner from './components/LoadingSpinner';

const HomePage = lazy(() => import('./pages/HomePage'));
const SignInPage = lazy(() => import('./pages/SignInPage'));
const SignUpPage = lazy(() => import('./pages/SignUpPage'));
const MainPage = lazy(() => import('./pages/MainPage'));
const AnalysisDashboardPage = lazy(() => import('./pages/AnalysisDashboardPage'));
const UploadPage = lazy(() => import('./pages/UploadPage'));
const UserProfilePage = lazy(() => import('./pages/UserProfilePage'));
const SpeechEvaluationDetailPage = lazy(() => import('./pages/SpeechEvaluationDetailPage'));
const NonverbalEvaluationDetailPage = lazy(() => import('./pages/NonverbalEvaluationDetailPage'));
const ScriptUpload = lazy(() => import('./pages/ScriptUpload'));

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        <Suspense fallback={<LoadingSpinner />}>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/signin" element={<SignInPage />} />
            <Route path="/signup" element={<SignUpPage />} />
            <Route path="/main" element={<MainPage />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/profile" element={<UserProfilePage />} />

            {/* Analysis pages */}
            <Route path="/analysis" element={<AnalysisDashboardPage />} />
            <Route path="/analysis/:presentationId" element={<AnalysisDashboardPage />} />

            {/* 대본 업로드 */}
            <Route path="/analysis/script-upload" element={<ScriptUpload />} />

            {/* Detail analysis pages */}
            <Route path="/analysis/:type/:presentationId" element={<SpeechEvaluationDetailPage />} />
            <Route path="/analysis/nonverbal/:presentationId" element={<NonverbalEvaluationDetailPage />} />
          </Routes>
        </Suspense>
      </Router>
    </ThemeProvider>
  );
}

export default App;