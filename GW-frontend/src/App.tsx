// src/App.tsx
import { ThemeProvider, CssBaseline } from '@mui/material';
import theme from './styles/theme'; // 경로는 너의 위치에 맞게 수정
import { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route} from 'react-router-dom';
// 로딩스피너(각 화면을 별도의 청크로 분리해 사용자가 방문하는 페이지만 로드하는 방식) 적용
import LoadingSpinner from './components/LoadingSpinner';
// 새로운 방식: lazy 로딩
const HomePage = lazy(() => import('./pages/HomePage'));
const SignInPage = lazy(() => import('./pages/SignInPage'));
const SignUpPage = lazy(() => import('./pages/SignUpPage'));
const MainPage = lazy(() => import('./pages/MainPage'));
const AnalysisDashboardPage = lazy(() => import('./pages/AnalysisDashboardPage'));
const UploadPage = lazy(() => import('./pages/UploadPage'));
const UserProfilePage = lazy(() => import('./pages/UserProfilePage'));
const SpeechEvaluationDetailPage = lazy(() => import('./pages/SpeechEvaluationDetailPage'));
const NonverbalEvaluationDetailPage = lazy(() => import('./pages/NonverbalEvaluationDetailPage'));

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
            <Route path="/analysis" element={<AnalysisDashboardPage />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/profile" element={<UserProfilePage />} />

            {/* Analysis pages */}
            <Route path="/analysis" element={<AnalysisDashboardPage />} />
            <Route path="/analysis/:presentationId" element={<AnalysisDashboardPage />} />

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
