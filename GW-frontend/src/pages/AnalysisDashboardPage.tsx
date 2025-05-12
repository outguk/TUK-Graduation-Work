// export default AnalysisDashboardPage;

/**
 * AnalysisDashboardPage.tsx
 * 
 * 백엔드 개발자 참고사항:
 * 이 파일은 발표 분석 대시보드의 메인 페이지입니다.
 * 좌측의 PresentationSidebar 컴포넌트를 통해 발표 목록을 표시하고,
 * 선택된 발표의 분석 데이터를 차트와 카드로 보여줍니다.
 * 
 * 필요한 API 엔드포인트:
 * 1. GET /spring/api/my-analyses - 사용자의 모든 발표 목록
 * 2. GET /spring/api/get-analysis?filename={filename} - 특정 발표의 상세 분석 데이터
 * 3. GET /spring/api/user/profile - 현재 로그인한 사용자 정보
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Container,
  Typography,
  Card,
  CardContent,
  Grid,
  Paper,
  Button,
  Fade,
  Divider,
  CircularProgress,
  Alert,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { useNavigate, useParams } from 'react-router-dom';
import {
  TextSnippet as TextSnippetIcon,
  GraphicEq as GraphicEqIcon,
  PersonOutline as PersonOutlineIcon,
} from '@mui/icons-material';
import axios from 'axios';
import PresentationSidebar from '../components/PresentationSidebar';

/**
 * 백엔드 개발자 참고사항:
 * 인터페이스 정의입니다. API 응답 형식과 일치해야 합니다.
 */
interface PresentationItem {
  id: string;        // 발표 고유 식별자 (filename)
  title: string;     // 발표 제목 (filename에서 확장자 제거)
  date: string;      // 발표 날짜 (YYYY.MM.DD)
  duration: string;  // 발표 길이 (M:SS, 기본값 제공)
}

interface UserProfile {
  name: string;      // 사용자 이름
}

interface PaceDataPoint {
  time: string;
  wpm: number;
}

interface VolumeDataPoint {
  time: string;
  db: number;
}

interface AnalysisData {
  _id: string;
  filename: string;
  speaking_speed?: { segment_wpm: Array<{ start: number; end: number; wpm: number }> };
  volume_analysis?: { segment_data: Array<{ time_stamps: [number, number]; db: number }> };
  speaking_evaluation?: { overall_score: number };
  volume_evaluation?: { overall_score: number };
  duration?: string;
}

const AnalysisDashboardPage: React.FC = () => {
  const navigate = useNavigate();
  const { presentationId } = useParams<{ presentationId?: string }>();

  /**
   * 백엔드 개발자 참고사항:
   * 상태 변수입니다.
   * - presentations: 발표 목록 (PresentationSidebar로 전달)
   * - userProfile: 사용자 정보 (PresentationSidebar로 전달)
   * - selectedPresentationId: 현재 선택된 발표 ID
   * - analysisData: 선택된 발표의 분석 데이터
   */
  const [presentations, setPresentations] = useState<PresentationItem[]>([]);
  const [userProfile, setUserProfile] = useState<UserProfile>({ name: '' });
  const [selectedPresentationId, setSelectedPresentationId] = useState<string | null>(presentationId || null);
  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null);
  const [paceData, setPaceData] = useState<PaceDataPoint[]>([]);
  const [volumeData, setVolumeData] = useState<VolumeDataPoint[]>([]);
  const [duration, setDuration] = useState("0:00");
  const [overallScore, setOverallScore] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showContent, setShowContent] = useState(false);


  // 정상 범위와 심각한 벗어남의 기준
  const SPEED_NORMAL_MIN = 100;
  const SPEED_NORMAL_MAX = 140;
  const SPEED_SEVERE_DEVIATION = 30;

  /**
   * 백엔드 개발자 참고사항:
   * 발표 팁 배열입니다. 5초마다 하나씩 표시됩니다.
   */
  const presentationTips = [
    "The core of a good presentation is clarity.",
    "Presenting is not about speaking, but connecting.",
    "A quiet beginning captures audience attention.",
    "Begin strong, end stronger.",
    "Eye contact strengthens connection with your audience.",
    "Tell stories, not just numbers.",
    "Deliver only three key messages your audience will remember.",
    "Strategic silence is a powerful presentation technique.",
    "Visual aids should complement your words, not replace them.",
    "Every audience member is silently asking 'Why should I care?'"
  ];

  const [currentTipIndex, setCurrentTipIndex] = useState(0);
  const [fadeTip, setFadeTip] = useState(true);

  /**
   * 백엔드 개발자 참고사항:
   * 페이지 로드 시 콘텐츠 페이드인 애니메이션과 팁 회전 설정
   */
  useEffect(() => {
    const timer = setTimeout(() => setShowContent(true), 300);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    const tipInterval = setInterval(() => {
      setFadeTip(false);
      setTimeout(() => {
        setCurrentTipIndex((prevIndex) => (prevIndex + 1) % presentationTips.length);
        setFadeTip(true);
      }, 500);
    }, 5000);
    return () => clearInterval(tipInterval);
  }, []);

  /**
   * 백엔드 개발자 참고사항:
   * 발표 목록과 사용자 정보를 가져옵니다.
   * - GET /spring/api/my-analyses: 발표 목록
   * - GET /spring/api/user/profile: 사용자 프로필
   * - 최초 로드 시 presentationId가 없으면 첫 번째 발표 선택
   */
  useEffect(() => {
    const fetchInitialData = async () => {
      setLoading(true);
      setError(null);

      try {
        const token = localStorage.getItem('token');
        if (!token) {
          throw new Error('인증 토큰이 없습니다.');
        }

        // 사용자 프로필 가져오기
        const profileResponse = await axios.get('/spring/api/user/profile', {
          headers: { Authorization: `Bearer ${token}` },
        });
        setUserProfile({ name: profileResponse.data.name });

        // 발표 목록 가져오기
        const analysesResponse = await axios.get('/spring/api/my-analyses', {
          headers: { Authorization: `Bearer ${token}` },
        });
        const presentationsData: PresentationItem[] = analysesResponse.data.analyses.map((item: any) => ({
          id: item.filename,
          title: item.filename.split('.')[0],
          date: new Date(item.timestamp).toLocaleDateString('ko-KR', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
          }).replace(/\//g, '.'),
          duration: '0:45', // 백엔드에서 제공 시 대체
        }));
        setPresentations(presentationsData);

        // presentationId가 없으면 첫 번째 발표 선택
        if (!selectedPresentationId && presentationsData.length > 0) {
          setSelectedPresentationId(presentationsData[0].id);
          navigate(`/analysis/${presentationsData[0].id}`, { replace: true });
        }
      } catch (err: any) {
        setError(err.response?.data?.error || '초기 데이터를 불러오는 중 오류가 발생했습니다.');
      } finally {
        setLoading(false);
      }
    };

    fetchInitialData();
  }, []);

  /**
   * 백엔드 개발자 참고사항:
   * 선택된 발표 ID가 변경될 때 분석 데이터를 가져옵니다.
   * - GET /spring/api/get-analysis?filename={filename}
   * - 응답 데이터를 차트와 카드에 표시할 형식으로 가공
   */
  useEffect(() => {
    const fetchAnalysisData = async () => {
      if (!selectedPresentationId) return;

      setLoading(true);
      setError(null);

      try {
        const token = localStorage.getItem('token');
        if (!token) {
          throw new Error('인증 토큰이 없습니다.');
        }

        const response = await axios.get('/spring/api/get-analysis', {
          params: { filename: selectedPresentationId },
          headers: { Authorization: `Bearer ${token}` },
        });

        const data: AnalysisData = response.data;
        setAnalysisData(data);

        // 발표 속도 데이터 가공
        const pace: PaceDataPoint[] = data.speaking_speed?.segment_wpm.map((segment) => ({
          time: formatTime(segment.start),
          wpm: segment.wpm,
        })) || [];
        setPaceData(pace);

        // 음량 데이터 가공
        const volume: VolumeDataPoint[] = data.volume_analysis?.segment_data.map((segment) => ({
          time: formatTime(segment.time_stamps[0]),
          db: segment.db,
        })) || [];
        setVolumeData(volume);

        // 발표 길이 계산
        const maxTime = Math.max(
          ...data.speaking_speed?.segment_wpm.map((s) => s.end) || [0],
          ...data.volume_analysis?.segment_data.map((s) => s.time_stamps[1]) || [0]
        );
        setDuration(formatTime(maxTime));

        // 전체 점수 계산
        const speakingScore = data.speaking_evaluation?.overall_score || 0;
        const volumeScore = data.volume_evaluation?.overall_score || 0;
        setOverallScore(Math.round((speakingScore + volumeScore) / 2));
      } catch (err: any) {
        console.error('분석 데이터 로드 에러:', err);
        setError(err.response?.data?.error || '데이터를 불러오는 중 오류가 발생했습니다.');
      } finally {
        setLoading(false);
      }
    };

    fetchAnalysisData();
  }, [selectedPresentationId, navigate]);

  /**
   * 백엔드 개발자 참고사항:
   * 시간을 MM:SS 형식으로 포맷팅하는 유틸리티 함수
   */
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  /**
   * 백엔드 개발자 참고사항:
   * PresentationSidebar에서 발표 선택 시 호출
   * URL을 업데이트하고 분석 데이터를 로드
   */
  const handleSelectPresentation = (id: string) => {
    setSelectedPresentationId(id);
    navigate(`/analysis/${id}`, { replace: true });
  };

  /**
   * 백엔드 개발자 참고사항:
   * 세부 분석 페이지로 이동하는 함수들
   * 선택된 발표 ID를 URL 파라미터로 전달
   */
  const handleNavigateToSpeed = () => {
    navigate(`/analysis/speed/${selectedPresentationId}`);
  };

  const handleNavigateToVolume = () => {
    navigate(`/analysis/volume/${selectedPresentationId}`);
  };

  const handleNavigateToScript = () => {
    navigate(`/analysis/script/${selectedPresentationId}`);
  };

  const handleNavigateToNonverbal = () => {
    navigate(`/analysis/nonverbal/${selectedPresentationId}`);
  };

  /**
   * 백엔드 개발자 참고사항:
   * 현재 발표 제목 표시
   * 분석 데이터가 있으면 filename에서 제목 추출
   */
  const currentPresentationTitle = analysisData?.filename.split('.')[0] || 'Presentation Analysis';

  return (
    <Box
      sx={{
        display: 'flex',
        minHeight: '100vh',
        background: '#FFFFFF',
      }}
    >
      {/* 
       * 백엔드 개발자 참고사항:
       * PresentationSidebar에 전달되는 props:
       * - presentations: 발표 목록 (API에서 가져옴)
       * - selectedPresentationId: 현재 선택된 발표 ID
       * - onSelectPresentation: 발표 선택 핸들러
       * - userProfile: 사용자 정보 (API에서 가져옴)
       */}
      <PresentationSidebar
        presentations={presentations}
        selectedPresentationId={selectedPresentationId}
        onSelectPresentation={handleSelectPresentation}
        userProfile={userProfile}
      />

      {/* Main content */}
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          py: { xs: 4, md: 6 },
          px: { xs: 2, md: 4 },
          marginLeft: '8%',
          transition: 'margin-left 0.3s ease',
          width: 'calc(100% - 8%)',
        }}
      >
        <Container maxWidth="xl">
          {/* Header section */}
          <Box sx={{ mb: 5 }}>
            <Typography 
              variant="h4" 
              component="h1"
              fontWeight={700}
              sx={{ 
                mb: 1,
                position: 'relative',
                display: 'inline-block'
              }}
            >
              {currentPresentationTitle}
            </Typography>
            <Typography 
              variant="body1" 
              color="text.secondary"
              sx={{ mt: 3 }}
            >
              영상 분석이 완료되었습니다! 결과를 확인해 보세요
            </Typography>
          </Box>
          
          <Fade in={showContent} timeout={1000}>
            <Box>
              {error && (
                <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
                  {error}
                </Alert>
              )}

              {loading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                  <CircularProgress />
                </Box>
              ) : !analysisData ? (
                <Box sx={{ textAlign: 'center', py: 4 }}>
                  <Typography variant="h6" color="text.secondary">
                    분석 데이터가 없습니다. 새 영상을 업로드해 주세요.
                  </Typography>
                  <Button
                    variant="contained"
                    onClick={() => navigate('/upload')}
                    sx={{ mt: 2 }}
                  >
                    업로드 페이지로 이동
                  </Button>
                </Box>
              ) : (
                <Grid container spacing={4}>
                  {/* Left column: Summary + Tips */}
                  <Grid item xs={12} md={3}>
                    {/* Summary card */}
                    <Card 
                      elevation={0}
                      sx={{ 
                        mb: 4, 
                        borderRadius: 3,
                        boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
                        height: '220px',
                        position: 'relative',
                        overflow: 'hidden',
                        '&::before': {
                          content: '""',
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          width: '100%',
                          height: '4px',
                          background: '#000'
                        }
                      }}
                    >
                      <CardContent sx={{ p: 3 }}>
                        <Typography 
                          variant="h6" 
                          component="h2"
                          fontWeight={600}
                          sx={{ mb: 3 }}
                        >
                          요약 정보
                        </Typography>
                        
                        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                          <Typography variant="body2" color="text.secondary" sx={{ width: '50%' }}>
                            총 영상 시간
                          </Typography>
                          <Typography variant="h5" fontWeight={700} sx={{ width: '50%', textAlign: 'right' }}>
                            {duration}
                          </Typography>
                        </Box>
                        
                        <Divider sx={{ my: 2 }} />
                        
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <Typography variant="body2" color="text.secondary" sx={{ width: '50%' }}>
                            Overall Score
                          </Typography>
                          <Typography 
                            variant="h4" 
                            fontWeight={700} 
                            sx={{ 
                              width: '50%', 
                              textAlign: 'right',
                              color: '#000'
                            }}
                          >
                            {overallScore}
                          </Typography>
                        </Box>
                      </CardContent>
                    </Card>
                    
                    {/* Tips card */}
                    <Card 
                      elevation={0}
                      sx={{ 
                        borderRadius: 3,
                        boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
                        height: 'calc(100% - 220px - 32px)',
                        minHeight: '200px',
                        display: 'flex',
                        flexDirection: 'column',
                        position: 'relative',
                        overflow: 'hidden',
                        '&::before': {
                          content: '""',
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          width: '100%',
                          height: '4px',
                          background: '#000'
                        }
                      }}
                    >
                      <CardContent sx={{ p: 3, flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
                        <Typography 
                          variant="h6" 
                          component="h2"
                          fontWeight={600}
                          sx={{ mb: 3 }}
                        >
                          발표 팁
                        </Typography>
                        
                        <Box 
                          sx={{ 
                            display: 'flex', 
                            flexDirection: 'column',
                            justifyContent: 'center',
                            alignItems: 'center',
                            flexGrow: 1,
                            px: 2
                          }}
                        >
                          <Fade in={fadeTip} timeout={500}>
                            <Typography 
                              variant="h6" 
                              align="center"
                              sx={{ 
                                fontWeight: 500,
                                fontStyle: 'italic',
                                lineHeight: 1.6
                              }}
                            >
                              "{presentationTips[currentTipIndex]}"
                            </Typography>
                          </Fade>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                  
                  {/* 중간 차트 (속도와 음량 부분) */}
                  <Grid item xs={12} md={6}>
                    {/* Speaking pace chart */}
                    <Paper 
                      elevation={0}
                      sx={{ 
                        p: 3, 
                        mb: 4, 
                        borderRadius: 3,
                        boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
                        cursor: 'pointer',
                        transition: 'all 0.3s ease',
                        '&:hover': {
                          boxShadow: '0 8px 30px rgba(0,0,0,0.12)',
                          transform: 'translateY(-4px)'
                        },
                        height: '48%',
                        position: 'relative',
                        overflow: 'hidden',
                        '&::before': {
                          content: '""',
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          width: '100%',
                          height: '4px',
                          background: '#000'
                        }
                      }}
                      onClick={handleNavigateToSpeed}
                      role="button"
                      tabIndex={0}
                      aria-label="Speaking pace analysis details"
                    >
                      <Typography variant="h6" component="h2" fontWeight={600}>
                        말하기 속도 (분당 단어 수)
                      </Typography>
                      
                      <Box
                        sx={{
                          position: 'relative',
                          width: '100%',
                          display: 'flex',
                          justifyContent: 'center',
                          alignItems: 'center',
                          mb: 2
                        }}
                      >
                        <Box
                          sx={{
                            position: 'relative',
                            width: '240px',
                            height: '160px',
                            margin: '0 auto',
                            mt: 2
                          }}
                        >
                          {/* Background gauge */}
                          <Box
                            component="svg"
                            viewBox="-5 -5 110 60"
                            sx={{
                              width: '100%',
                              height: '100%',
                              position: 'absolute',
                              top: 0,
                              left: 0,
                              transform: 'translateY(-10px) scale(1.3)'
                            }}
                          >
                            <path
                              d="M 10,50 A 40,40 0 0,1 90,50"
                              fill="none"
                              stroke="#f5f5f5"
                              strokeWidth="12"
                              strokeLinecap="butt"
                            />
                            <text x="0" y="20" fontSize="4" fill="#666" textAnchor="middle">
                              0
                            </text>
                            <text x="50" y="55" fontSize="4" fill="#666" textAnchor="middle">
                              100
                            </text>
                            <text x="100" y="20" fontSize="4" fill="#666" textAnchor="middle">
                              200
                            </text>
                          </Box>

                          {/* Active gauge */}
                          <Box
                            component="svg"
                            viewBox="-5 -5 110 60"
                            sx={{
                              width: '100%',
                              height: '100%',
                              position: 'absolute',
                              top: 0,
                              left: 0,
                              transform: 'translateY(-10px) scale(1.3)'
                            }}
                          >
                            <path
                              d={`M 10,50 A 40,40 0 0,1 ${(() => {
                                const avgWpm = paceData.reduce((sum, item) => sum + item.wpm, 0) / (paceData.length || 1);
                                const percentage = Math.min(Math.max(avgWpm / 200, 0), 1);
                                const theta = (180 - 180 * percentage) * (Math.PI / 180);
                                const cx = 50;
                                const cy = 50;
                                const r = 40;
                                const x = cx + r * Math.cos(theta);
                                const y = cy - r * Math.sin(theta);
                                return `${x},${y}`;
                              })()}`}
                              fill="none"
                              stroke={(() => {
                                // Calculate average WPM
                                const avgWpm = paceData.reduce((sum, item) => sum + item.wpm, 0) / (paceData.length || 1);
                                
                                // Determine color based on criteria
                                if (avgWpm >= SPEED_NORMAL_MIN && avgWpm <= SPEED_NORMAL_MAX) {
                                  return "#4caf50"; // Green - 정상 범위
                                } else if (avgWpm < SPEED_NORMAL_MIN - SPEED_SEVERE_DEVIATION || 
                                          avgWpm > SPEED_NORMAL_MAX + SPEED_SEVERE_DEVIATION) {
                                  return "#e53935"; // Red - 위험 범위 (30+ WPM off)
                                } else {
                                  return "#ff8c00"; // Orange - 경고 범위 (within 30 WPM)
                                }
                              })()}
                              strokeWidth="12"
                              strokeLinecap="butt"
                            />
                          </Box>

                          {/* Center text */}
                          <Box
                            sx={{
                              position: 'absolute',
                              top: '60px',
                              left: 0,
                              width: '100%',
                              textAlign: 'center'
                            }}
                          >
                            <Typography
                              variant="h2"
                              component="div"
                              fontWeight="700"
                              sx={{ lineHeight: 1 }}
                            >
                              {Math.round(
                                paceData.reduce((sum, item) => sum + item.wpm, 0) / (paceData.length || 1)
                              )}
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              words/min
                            </Typography>
                          </Box>

                          {/* Range labels */}
                          <Box
                            sx={{
                              position: 'absolute',
                              bottom: '-10px',
                              width: '110%',
                              display: 'flex',
                              justifyContent: 'space-between',
                              px: 2
                            }}
                          >
                            <Typography variant="caption" color="text.secondary" sx={{ ml: -3 }}>
                              slow
                            </Typography>
                            <Typography variant="caption" color="text.secondary">
                              fast
                            </Typography>
                          </Box>
                        </Box>
                      </Box>

                      {/* Pace evaluation message */}
                      <Typography 
                        variant="body1" 
                        sx={{ 
                          textAlign: 'center',
                          fontWeight: 500,
                          mb: 4,
                          mt: 2,
                          fontSize: '1rem'
                        }}
                      >
                        {(() => {
                          const avgWpm = paceData.reduce((sum, item) => sum + item.wpm, 0) / (paceData.length || 1);
                          if (avgWpm < 100) return "말하기 속도가 조금 느려요. 조금 더 속도감 있게 말하려고 노력해보세요.";
                          if (avgWpm > 150) return "말하기 속도가 조금 빨라요. 조금 더 차분하게 말하려고 노력해보세요.";
                          return "말하기 속도가 적절합니다. 이 속도를 유지하세요!";
                        })()}
                      </Typography>
                    </Paper>
                    
                    {/* Volume chart */}
                    <Paper 
                      elevation={0}
                      sx={{ 
                        p: 3, 
                        borderRadius: 3,
                        boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
                        cursor: 'pointer',
                        transition: 'all 0.3s ease',
                        '&:hover': {
                          boxShadow: '0 8px 30px rgba(0,0,0,0.12)',
                          transform: 'translateY(-4px)'
                        },
                        height: '48%',
                        position: 'relative',
                        overflow: 'hidden',
                        '&::before': {
                          content: '""',
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          width: '100%',
                          height: '4px',
                          background: '#000'
                        }
                      }}
                      onClick={handleNavigateToVolume}
                      role="button"
                      tabIndex={0}
                      aria-label="Volume analysis details"
                    >
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                        <Typography variant="h6" component="h2" fontWeight={600}>
                          음량 (dB)
                        </Typography>
                        <GraphicEqIcon />
                      </Box>
                      
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                        해당 발표 평균 음량: {volumeData.length 
                          ? (volumeData.reduce((sum, item) => sum + item.db, 0) / volumeData.length).toFixed(1) 
                          : "0"} dB (적정 음량 범위: 60-75 dB)
                      </Typography>
                      
                      <Box sx={{ height: '200px', width: '100%' }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={volumeData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" vertical={false} />
                            <XAxis dataKey="time" />
                            <YAxis domain={[30, 90]} />
                            <Tooltip 
                              // formatter={(db) => [`${db} dB`, '음량']}
                              contentStyle={{ 
                                backgroundColor: '#fff', 
                                borderRadius: '8px',
                                boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                                border: 'none' 
                              }} 
                            />
                            <ReferenceLine y={60} stroke="#999" strokeDasharray="3 3" />
                            <ReferenceLine y={70} stroke="#999" strokeDasharray="3 3" />
                            <Line 
                              type="monotone" 
                              dataKey="db" 
                              stroke="#000" 
                              strokeWidth={3} 
                              dot={{ r: 4 }}
                              activeDot={{ r: 6, strokeWidth: 1, stroke: '#FFF' }}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      </Box>
                    </Paper>
                  </Grid>
                  
                  {/* Right column: Detail links */}
                  <Grid item xs={12} md={3}>
                    {/* Script analysis card */}
                    <Card 
                      elevation={0}
                      sx={{ 
                        mb: 4, 
                        borderRadius: 3,
                        boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
                        height: '48%',
                        display: 'flex',
                        flexDirection: 'column',
                        cursor: 'pointer',
                        transition: 'all 0.3s ease',
                        '&:hover': {
                          boxShadow: '0 8px 30px rgba(0,0,0,0.12)',
                          transform: 'translateY(-4px)'
                        },
                        position: 'relative',
                        overflow: 'hidden',
                        '&::before': {
                          content: '""',
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          width: '100%',
                          height: '4px',
                          background: '#000'
                        }
                      }}
                      onClick={handleNavigateToScript}
                      role="button"
                      tabIndex={0}
                      aria-label="Script analysis details"
                    >
                      <CardContent sx={{ p: 4, flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
                        <Box 
                          sx={{ 
                            mb: 3,
                            display: 'flex',
                            justifyContent: 'center',
                            alignItems: 'center',
                            width: '80px',
                            height: '80px',
                            borderRadius: '50%',
                            background: 'rgba(0, 0, 0, 0.05)',
                            mx: 'auto',
                            transition: 'all 0.3s ease'
                          }}
                        >
                          <TextSnippetIcon sx={{ fontSize: 40, color: '#000' }} />
                        </Box>
                        
                        <Typography 
                          variant="h5" 
                          component="h2" 
                          align="center"
                          fontWeight={600}
                          sx={{ mb: 2 }}
                        >
                          대본 분석 결과
                        </Typography>
                        
                        <Typography 
                          variant="body1" 
                          align="center"
                          color="text.secondary"
                          sx={{ mb: 3 }}
                        >
                          발표 대본과 문장에 대한 분석을 확인하세요.
                        </Typography>
                        
                        <Box sx={{ mt: 'auto', display: 'flex', justifyContent: 'center' }}>
                          <Button 
                            variant="outlined" 
                            color="primary"
                            size="large"
                            sx={{ 
                              borderRadius: 8,
                              px: 3,
                              py: 1,
                              borderColor: '#000',
                              color: '#000',
                              '&:hover': {
                                borderColor: '#333',
                                backgroundColor: 'rgba(0, 0, 0, 0.04)'
                              }
                            }}
                          >
                            세부 결과 보기
                          </Button>
                        </Box>
                      </CardContent>
                    </Card>
                    
                    {/* Nonverbal analysis card */}
                    <Card 
                      elevation={0}
                      sx={{ 
                        borderRadius: 3,
                        boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
                        height: '48%',
                        display: 'flex',
                        flexDirection: 'column',
                        cursor: 'pointer',
                        transition: 'all 0.3s ease',
                        '&:hover': {
                          boxShadow: '0 8px 30px rgba(0,0,0,0.12)',
                          transform: 'translateY(-4px)'
                        },
                        position: 'relative',
                        overflow: 'hidden',
                        '&::before': {
                          content: '""',
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          width: '100%',
                          height: '4px',
                          background: '#000'
                        }
                      }}
                      onClick={handleNavigateToNonverbal}
                      role="button"
                      tabIndex={0}
                      aria-label="Nonverbal analysis details"
                    >
                      <CardContent sx={{ p: 4, flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
                        <Box 
                          sx={{ 
                            mb: 3,
                            display: 'flex',
                            justifyContent: 'center',
                            alignItems: 'center',
                            width: '80px',
                            height: '80px',
                            borderRadius: '50%',
                            background: 'rgba(0, 0, 0, 0.05)',
                            mx: 'auto',
                            transition: 'all 0.3s ease'
                          }}
                        >
                          <PersonOutlineIcon sx={{ fontSize: 40, color: '#000' }} />
                        </Box>
                        
                        <Typography 
                          variant="h5" 
                          component="h2" 
                          align="center"
                          fontWeight={600}
                          sx={{ mb: 2 }}
                        >
                          비언어 요소 분석 결과
                        </Typography>
                        
                        <Typography 
                          variant="body1" 
                          align="center"
                          color="text.secondary"
                          sx={{ mb: 3 }}
                        >
                          자세, 머리동작, 제스처 및 기타 비언어적 요소에 대한 분석을 확인하세요.
                        </Typography>
                        
                        <Box sx={{ mt: 'auto', display: 'flex', justifyContent: 'center' }}>
                          <Button 
                            variant="outlined" 
                            color="primary"
                            size="large"
                            sx={{ 
                              borderRadius: 8,
                              px: 3,
                              py: 1,
                              borderColor: '#000',
                              color: '#000',
                              '&:hover': {
                                borderColor: '#333',
                                backgroundColor: 'rgba(0, 0, 0, 0.04)'
                              }
                            }}
                          >
                            세부 결과 보기
                          </Button>
                        </Box>
                      </CardContent>
                    </Card>
                  </Grid>
                </Grid>
              )}
            </Box>
          </Fade>
          
          {/* Footer */}
          <Box 
            component="footer"
            sx={{ 
              mt: 8, 
              textAlign: 'center', 
              borderTop: '1px solid #eee',
              pt: 4,
              pb: 2,
              color: '#666'
            }}
          >
            <Typography variant="body2">
              © 2025 PRESENT INSIGHT. All rights reserved.
            </Typography>
          </Box>
        </Container>
      </Box>
    </Box>
  );
};

export default AnalysisDashboardPage;