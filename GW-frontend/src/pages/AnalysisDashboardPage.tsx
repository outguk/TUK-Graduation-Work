/**
 * AnalysisDashboardPage.tsx
 * 
 * 개요: 
 * 이 컴포넌트는 사용자의 발표 분석 결과를 종합적으로 보여주는 대시보드 페이지입니다.
 * 
 * 백엔드 연동 개요:
 * - 사용자가 업로드한 발표 영상의 분석 결과를 종합하여 보여줍니다.
 * - 이 페이지는 말하기 속도(WPM), 음량(dB), 비언어적 요소 분석 등의 데이터를 시각화합니다.
 * - 현재는 더미 데이터를 사용하고 있지만, 추후 백엔드 API와 연동되어야 합니다.
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
import { useNavigate } from 'react-router-dom';
import {
  TextSnippet as TextSnippetIcon,
  GraphicEq as GraphicEqIcon,
  PersonOutline as PersonOutlineIcon,
} from '@mui/icons-material';

const AnalysisDashboardPage: React.FC = () => {
  const navigate = useNavigate();
  
  // 애니메이션 상태 관리
  const [showContent, setShowContent] = useState(false);
  
  /**
   * 말하기 속도(WPM) 데이터
   * 
   * 백엔드 연동 시:
   * 1. API 엔드포인트에서 발표 분석 데이터를 가져와야 합니다.
   * 2. 엔드포인트 예시: GET /api/analysis/:presentationId 또는 GET /api/analysis/latest
   * 3. 응답 데이터에서 speaking_speed 필드를 추출하여 차트 데이터 형식으로 변환해야 합니다.
   * 
   * 응답 예상 형식:
   * {
   *   speaking_speed: {
   *     "00:00-01:00": 115, 이때 각 구간의 평균 시간값을 가져오거나(전체 구간별) 나눌 구간의 개수 정해서 그 개수만큼의 구간만 가져오고 해당 구간의 평균 wpm을 넘겨줄지 생각해봐야 할듯
   *     "01:00-02:00": 125,
   *     ...
   *   }
   * }
   * 
   * 현재는 더미 데이터로 구현되어 있습니다.
   */
  const paceData = [
    { time: '00:00', wpm: 110 },
    { time: '01:00', wpm: 125 },
    { time: '02:00', wpm: 140 },
    { time: '03:00', wpm: 130 },
    { time: '04:00', wpm: 121 },
    { time: '05:00', wpm: 115 },
    { time: '06:00', wpm: 105 },
    { time: '07:00', wpm: 120 },
  ];
  
  /**
   * 음량(dB) 데이터
   * 
   * 백엔드 연동 시:
   * 1. 동일한 API 엔드포인트에서 음량 데이터도 함께 가져와야 합니다.
   * 2. 응답 데이터에서 volume_analysis 필드를 추출하여 차트 데이터 형식으로 변환해야 합니다.
   * 
   * 응답 예상 형식:
   * {
   *   volume_analysis: {
   *     "00:00-01:00": 65, 여기도 이때 각 구간의 평균 시간값을 가져오거나(전체 구간별) 나눌 구간의 개수 정해서 그 개수만큼의 구간만 가져오고 해당 구간의 평균 wpm을 넘겨줄지 생각해봐야 할듯
   *     "01:00-02:00": 70,
   *     ...
   *   }
   * }
   */
  const volumeData = [
    { time: '00:00', db: 65 },
    { time: '01:00', db: 70 },
    { time: '02:00', db: 72 },
    { time: '03:00', db: 65 },
    { time: '04:00', db: 60 },
    { time: '05:00', db: 68 },
    { time: '06:00', db: 72 },
    { time: '07:00', db: 75 },
  ];
  
  /**
   * 발표 팁/명언 데이터
   * 
   * 백엔드 연동 시:
   * 1. 이 부분은 선택적으로 백엔드에서 가져올 수 있습니다.
   * 2. 엔드포인트 예시: GET /api/presentation-tips
   * 3. 또는 프론트엔드에 하드코딩된 상태로 유지할 수도 있습니다.
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
  
  // 현재 표시되는 팁 인덱스와 페이드 애니메이션 상태
  const [currentTipIndex, setCurrentTipIndex] = useState(0);
  const [fadeTip, setFadeTip] = useState(true);
  
  // 페이지 로드 시 애니메이션 초기화
  useEffect(() => {
    const timer = setTimeout(() => setShowContent(true), 300);
    return () => clearTimeout(timer);
  }, []);
  
  // 5초마다 팁 순환 (애니메이션 포함)
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
   * 내비게이션 핸들러 함수들
   * 
   * 각 분석 상세 페이지로 이동하는 함수입니다.
   * URL 파라미터를 통해 특정 발표 ID를 전달할 수도 있습니다.
   * 예: navigate(`/analysis/speed/${presentationId}`)
   */
  const handleNavigateToSpeed = () => {
    // TODO: 백엔드 연동 시 현재 보고 있는 발표 ID를 URL에 포함하여 전달 각 유저별 다른 화면을 위해. 다른 방법이 있다면 고려해봐도 좋을 듯
    // 예: navigate(`/analysis/speed/${presentationId}`); ---------------------------------------------------------------------
    navigate('/analysis/speed');
  };
  
  const handleNavigateToVolume = () => {
    // TODO: 백엔드 연동 시 현재 보고 있는 발표 ID를 URL에 포함하여 전달
    navigate('/analysis/volume');
  };
  
  const handleNavigateToScript = () => {
    // TODO: 백엔드 연동 시 현재 보고 있는 발표 ID를 URL에 포함하여 전달
    navigate('/analysis/script');
  };
  
  const handleNavigateToNonverbal = () => {
    // TODO: 백엔드 연동 시 현재 보고 있는 발표 ID를 URL에 포함하여 전달
    navigate('/analysis/nonverbal');
  };

  /**
   * 백엔드 연동을 위한 데이터 로딩 부분
   * 
   * 현재는 구현되어 있지 않지만, 백엔드 연동 시 아래와 같은 useEffect를 추가해야 합니다.
   */
  /*
  // 발표 분석 데이터 가져오기
  useEffect(() => {
    const fetchAnalysisData = async () => {
      // 로딩 상태 설정
      setLoading(true);
      
      try {
        // API 호출
        // presentationId는 URL 파라미터나 상태로 관리
        // 또는 가장 최근 발표 데이터를 가져오는 엔드포인트 호출
        const response = await axios.get('/api/analysis/latest');
        
        // 응답 데이터 처리
        const data = response.data;
        
        // 요약 정보 설정
        // 예: setDuration(data.duration);
        //     setOverallScore(data.overall_score);
        
        // 차트 데이터 변환 및 설정
        // 예: setPaceData(formatPaceData(data.speaking_speed));
        //     setVolumeData(formatVolumeData(data.volume_analysis));
        
      } catch (error) {
        // 에러 처리
        console.error('데이터 로딩 오류:', error);
        // 에러 상태 설정 (필요시)
        // setError('데이터를 불러오는 중 오류가 발생했습니다.');
      } finally {
        // 로딩 상태 해제
        setLoading(false);
      }
    };
    
    fetchAnalysisData();
  }, []);
  */

  return (
    <Box
      component="main"
      sx={{
        minHeight: '100vh',
        py: { xs: 4, md: 6 },
        px: { xs: 2, md: 4 },
        background: '#FFFFFF',
      }}
    >
      <Container maxWidth="xl">
        {/* 헤더 섹션 - 페이지 제목 및 설명 */}
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
            {/* TODO: 백엔드 연동 시 사용자 이름 또는 발표 제목을 동적으로 표시할 수 있음 */}
            Presentation Analysis
          </Typography>
          <Typography 
            variant="body1" 
            color="text.secondary"
            sx={{ mt: 3 }}
          >
            {/* TODO: 백엔드 연동 시 발표 날짜 등의 정보를 동적으로 표시할 수 있음 */}
            Your presentation video analysis is complete. Review your results below.
          </Typography>
        </Box>
        
        <Fade in={showContent} timeout={1000}>
          <Grid container spacing={4}>
            {/* 왼쪽 컬럼: 요약 정보 + 발표 팁 */}
            <Grid item xs={12} md={3}>
              {/* 요약 정보 카드 */}
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
                  
                  {/* 
                    총 발표 시간 - 백엔드 연동 시 아래 하드코딩된 값을 대체해야 함
                    API 응답에서 duration 필드 또는 유사한 필드를 사용
                  */}
                  <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                    <Typography variant="body2" color="text.secondary" sx={{ width: '50%' }}>
                      Total Duration
                    </Typography>
                    <Typography variant="h5" fontWeight={700} sx={{ width: '50%', textAlign: 'right' }}>
                      {/* TODO: 백엔드 연동 시 실제 발표 시간으로 대체 */}
                      7:45
                    </Typography>
                  </Box>
                  
                  <Divider sx={{ my: 2 }} />
                  
                  {/* 
                    전체 점수 - 백엔드 연동 시 아래 하드코딩된 값을 대체해야 함
                    API 응답에서 overall_score 필드 또는 유사한 필드를 사용
                  */}
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
                      {/* TODO: 백엔드 연동 시 실제 종합 점수로 대체 */}
                      85
                    </Typography>
                  </Box>
                </CardContent>
              </Card>
              
              {/* 발표 팁 카드 */}
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
                  
                  {/* 
                    발표 팁 표시 부분
                    백엔드 연동 시 팁 데이터를 API에서 가져올 수도 있음
                  */}
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
            
            {/* 중앙 컬럼: 차트 */}
            <Grid item xs={12} md={6}>
              {/* 
                말하기 속도 차트 
                백엔드 연동 시 paceData를 API 응답 데이터로 대체
              */}
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
                
                {/* 
                  게이지 차트 컨테이너
                  백엔드 연동 시 avgWpm 계산 부분을 API 응답 데이터로 대체
                */}
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
                      width: '240px',    // 전체 컨테이너 너비
                      height: '160px',   // 전체 컨테이너 높이 (상단에 게이지 반원, 중앙 텍스트가 들어감)
                      margin: '0 auto',  // 중앙 정렬
                      mt: 2
                    }}
                  >
                    {/* 배경 게이지 (회색) + 눈금 */}
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
                      {/* 위쪽 반원 (회색 배경) */}
                      <path
                        d="M 10,50 A 40,40 0 0,1 90,50"
                        fill="none"
                        stroke="#f5f5f5"
                        strokeWidth="12"
                        strokeLinecap="butt"
                      />

                      {/* 눈금 레이블 (예: 0 / 100 / 200) - 필요시 위치 조정 */}
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

                    {/* 
                      활성 게이지 (초록색)
                      백엔드 연동 시 아래 avgWpm 계산 부분을 API 응답 데이터로 대체
                    */}
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
                          // 예시: paceData로부터 평균 wpm 추출
                          // TODO: 백엔드 연동 시 API 응답에서 평균 WPM 값을 직접 사용하거나 계산
                          const avgWpm = paceData.reduce((sum, item) => sum + item.wpm, 0) / paceData.length;

                          // 0~200 범위를 [0~1]로 정규화
                          const percentage = Math.min(Math.max(avgWpm / 200, 0), 1);

                          // 180도(왼쪽) ~ 0도(오른쪽)에 해당: (180 - 180 * percentage)
                          // 각도 → 라디안 변환
                          const theta = (180 - 180 * percentage) * (Math.PI / 180);

                          // 중심 (50,50), 반경 40
                          const cx = 50;
                          const cy = 50;
                          const r = 40;

                          // (cx,cy) 기준 반원 위 점 좌표
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

                    {/* 
                      중앙 텍스트 (숫자, 단위)
                      백엔드 연동 시 아래 평균 WPM 계산 부분을 API 응답 데이터로 대체
                    */}
                    <Box
                      sx={{
                        position: 'absolute',
                        top: '60px',      // 게이지와 겹치지 않도록 적절히 조정
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
                        {/* TODO: 백엔드 연동 시 API 응답에서 평균 WPM 값을 직접 사용 */}
                        {Math.round(
                          paceData.reduce((sum, item) => sum + item.wpm, 0) / paceData.length
                        )}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        words/min
                      </Typography>
                    </Box>

                    {/* 속도 범위 라벨 (slow / fast) - 게이지 아래쪽에 배치 */}
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

                
                {/* 
                  속도 평가 메시지
                  백엔드 연동 시 이 함수에서 계산하는 부분을 API 응답에서 받은 평가 메시지로 대체 가능
                  또는 백엔드에서 평균 WPM 값만 받아와서 프론트엔드에서 메시지를 결정할 수도 있음
                */}
                <Typography 
                  variant="body1" 
                  sx={{ 
                    textAlign: 'center',
                    fontWeight: 500,
                    mb: 4,
                    mt: 2,
                    fontSize: '1rem' // 크기 증가
                  }}
                >
                  {(() => {
                    // TODO: 백엔드 연동 시 API 응답에서 평균 WPM을 받아 평가 메시지 생성
                    // 또는 백엔드에서 평가 메시지를 직접 받아올 수도 있음
                    const avgWpm = paceData.reduce((sum, item) => sum + item.wpm, 0) / paceData.length;
                    if (avgWpm < 100) return "Your pace is a bit slow. Try to speak slightly faster.";
                    if (avgWpm > 150) return "Your pace is a bit fast. Try to slow down slightly.";
                    return "Your pace is just right. Keep it up!";
                  })()}
                </Typography>
              </Paper>
              
              {/* 
                음량 차트
                백엔드 연동 시 volumeData를 API 응답 데이터로 대체
              */}
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
                
                {/* 
                  평균 음량 정보
                  백엔드 연동 시 평균 음량을 API 응답에서 받아와야 함
                */}
                <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                  {/* TODO: 백엔드 연동 시 실제 평균 음량 값으로 대체 */}
                  Average volume: 68.4 dB (Optimal range: 60-75 dB)
                </Typography>
                
                {/* 
                  음량 라인 차트 
                  백엔드 연동 시 volumeData를 API 응답 데이터로 대체
                */}
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
            
            {/* 오른쪽 컬럼: 상세 분석 링크 */}
            <Grid item xs={12} md={3}>
              {/* 
                스크립트 분석 링크 카드
                백엔드 연동 시 이 카드는 API에서 받아온 스크립트 분석 데이터로 연결됨
              */}
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
                    {/* 
                      스크립트 분석 설명
                      백엔드 연동 시, API 응답에 포함된 요약 정보나 메트릭을 여기에 표시 가능
                    */}
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
              
              {/* 
                비언어적 분석 링크 카드
                백엔드 연동 시 이 카드는 API에서 받아온 비언어적 요소 분석 데이터로 연결됨
              */}
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
                    {/* 
                      비언어적 분석 설명
                      백엔드 연동 시, API 응답에 포함된 요약 정보나 메트릭을 여기에 표시 가능
                      UploadPage.tsx의 nonverbal_analysis 필드와 연결 가능
                    */}
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
        
        {/* 푸터 */}
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
            {/* 
              저작권 정보
              필요에 따라 동적 연도 표시 가능
            */}
            © 2025 PRESENT INSIGHT. All rights reserved.
          </Typography>
        </Box>
      </Container>
    </Box>
  );
};

export default AnalysisDashboardPage;