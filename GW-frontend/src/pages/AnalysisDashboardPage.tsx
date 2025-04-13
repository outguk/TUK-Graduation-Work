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

// Import the sidebar component
import PresentationSidebar from '../components/PresentationSidebar';


/**
 * 백엔드 개발자 참고사항:
 * 현재는 목업 데이터를 사용하고 있지만, 실제 구현 시 이 부분을
 * 백엔드 API에서 데이터를 가져오는 로직으로 변경해야 합니다.
 * mockPresentations, mockAnalysisDataMap은 실제 API 응답으로 대체되어야 합니다.
 * 
 * 필요한 API 엔드포인트:
 * 1. GET /api/presentations - 사용자의 모든 발표 목록
 * 2. GET /api/presentations/:id - 특정 발표의 상세 분석 데이터
 */
// Import mock data
import { 
  mockPresentations, 
  mockAnalysisDataMap, 
  getPaceChartData, 
  getVolumeChartData 
} from '../components/mockAnalysisData';

const AnalysisDashboardPage: React.FC = () => {
  const navigate = useNavigate();

  /**
   * 백엔드 개발자 참고사항:
   * URL에서 presentationId 파라미터를 추출합니다.
   * 이 ID를 기반으로 특정 발표의 분석 데이터를 가져옵니다.
   */
  const { presentationId } = useParams<{ presentationId?: string }>();
  
  /**
   * 백엔드 개발자 참고사항:
   * 선택된 발표 ID 상태입니다. 
   * URL에서 파라미터가 없으면 첫 번째 발표를 기본값으로 사용합니다.
   * 실제 구현 시, 사용자의 최신 발표를 기본값으로 설정하는 것이 좋습니다.
   */
  const [selectedPresentationId, setSelectedPresentationId] = useState<string | null>(
    presentationId || mockPresentations[0]?.id || null
  );
  
  // State for current analysis data
  const [paceData, setPaceData] = useState<any[]>([]);
  const [volumeData, setVolumeData] = useState<any[]>([]);
  const [duration, setDuration] = useState("0:00");
  const [overallScore, setOverallScore] = useState(0);
  
  /**
   * 백엔드 개발자 참고사항:
   * 사이드바에서 발표를 선택했을 때 호출되는 함수입니다.
   * 선택된 발표 ID로 URL을 업데이트하고, 새 데이터를 로드합니다.
   */
  const handleSelectPresentation = (id: string) => {
    setSelectedPresentationId(id);
    // Update URL without page reload
    navigate(`/analysis/${id}`, { replace: true });
  };
  
  /**
   * 백엔드 개발자 참고사항:
   * 선택된 발표 ID가 변경될 때마다 해당 발표의 분석 데이터를 로드합니다.
   * 
   * 실제 구현 시에는 아래 부분을 API 호출로 대체해야 합니다:
   * const response = await fetch(`/api/presentations/${selectedPresentationId}`);
   * const analysisData = await response.json();
   * 
   * 응답 형식은 mockAnalysisDataMap의 객체 구조와 일치해야 합니다:
  */
  useEffect(() => {
    if (selectedPresentationId) {
      const analysisData = mockAnalysisDataMap[selectedPresentationId];
      
      if (analysisData) {
        // Use utility functions to format data for charts
        setPaceData(getPaceChartData(analysisData));
        setVolumeData(getVolumeChartData(analysisData));
        
        // Set presentation details
        setDuration(analysisData.duration);
        
        // Calculate overall score (average of speaking and volume scores)
        const speakingScore = analysisData.speaking_evaluation?.overall_score || 0;
        const volumeScore = analysisData.volume_evaluation?.overall_score || 0;
        setOverallScore(Math.round((speakingScore + volumeScore) / 2));
      }
    }
  }, [selectedPresentationId]);
  
  // Animation states
  const [showContent, setShowContent] = useState(false);
  
  // Presentation tips
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
  
  // Tips rotation state
  const [currentTipIndex, setCurrentTipIndex] = useState(0);
  const [fadeTip, setFadeTip] = useState(true);
  
  // Initialize animations on page load
  useEffect(() => {
    const timer = setTimeout(() => setShowContent(true), 300);
    return () => clearTimeout(timer);
  }, []);
  
  // Rotate tips every 5 seconds
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
   * 다음 함수들은 세부 분석 페이지로 이동하는 함수들입니다.
   * 현재 선택된 발표 ID를 URL 파라미터로 전달합니다.
   * 각 페이지에서는 이 ID를 사용하여 해당 발표의 세부 분석 데이터를 가져옵니다.
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
   * 현재 선택된 발표의 제목을 가져오는 부분입니다.
   * 실제 API에서는 발표 목록 또는 특정 발표 데이터에서 제목을 가져와야 합니다.
   */
  const currentPresentationTitle = selectedPresentationId
    ? mockPresentations.find(p => p.id === selectedPresentationId)?.title || "Presentation Analysis"
    : "Presentation Analysis";

  return (
    <Box
      sx={{
        display: 'flex',
        minHeight: '100vh',
        background: '#FFFFFF',
      }}
    >
      {/* Sidebar component with presentations */}
      <PresentationSidebar
        presentations={mockPresentations}
        selectedPresentationId={selectedPresentationId}
        onSelectPresentation={handleSelectPresentation}
        userProfile={{ name: "홍길동" }}
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
              Your presentation video analysis is complete. Review your results below.
            </Typography>
          </Box>
          
          <Fade in={showContent} timeout={1000}>
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
                      Summary
                    </Typography>
                    
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                      <Typography variant="body2" color="text.secondary" sx={{ width: '50%' }}>
                        Total Duration
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
                      Presentation Tips
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
              
              {/* Middle column: Charts */}
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
                      Present Speed (WPM)
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
                            // Calculate average WPM
                            const avgWpm = paceData.reduce((sum, item) => sum + item.wpm, 0) / (paceData.length || 1);

                            // Normalize to 0-1 range
                            const percentage = Math.min(Math.max(avgWpm / 200, 0), 1);

                            // Convert to angle (180° left to 0° right)
                            const theta = (180 - 180 * percentage) * (Math.PI / 180);

                            // Calculate point on arc
                            const cx = 50;
                            const cy = 50;
                            const r = 40;
                            const x = cx + r * Math.cos(theta);
                            const y = cy - r * Math.sin(theta);

                            return `${x},${y}`;
                          })()}`}
                          fill="none"
                          stroke="#AAD500"
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
                      if (avgWpm < 100) return "Your pace is a bit slow. Try to speak slightly faster.";
                      if (avgWpm > 150) return "Your pace is a bit fast. Try to slow down slightly.";
                      return "Your pace is just right. Keep it up!";
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
                      Volume Analysis (dB)
                    </Typography>
                    <GraphicEqIcon />
                  </Box>
                  
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                    Average volume: {volumeData.length 
                      ? (volumeData.reduce((sum, item) => sum + item.db, 0) / volumeData.length).toFixed(1) 
                      : "0"} dB (Optimal range: 60-75 dB)
                  </Typography>
                  
                  <Box sx={{ height: '200px', width: '100%' }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={volumeData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} />
                        <XAxis dataKey="time" />
                        <YAxis domain={[50, 85]} />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: '#fff', 
                            borderRadius: '8px',
                            boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                            border: 'none' 
                          }} 
                        />
                        <ReferenceLine y={60} stroke="#AAAAAA" strokeDasharray="3 3" />
                        <ReferenceLine y={75} stroke="#AAAAAA" strokeDasharray="3 3" />
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
                      Script Analysis
                    </Typography>
                    
                    <Typography 
                      variant="body1" 
                      align="center"
                      color="text.secondary"
                      sx={{ mb: 3 }}
                    >
                      Review analysis of your presentation script and verbal expressions.
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
                        View Details
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
                      Nonverbal Analysis
                    </Typography>
                    
                    <Typography 
                      variant="body1" 
                      align="center"
                      color="text.secondary"
                      sx={{ mb: 3 }}
                    >
                      Review analysis of your posture, eye contact, gestures, and other nonverbal elements.
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
                        View Details
                      </Button>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
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