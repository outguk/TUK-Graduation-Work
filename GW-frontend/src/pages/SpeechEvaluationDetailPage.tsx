/**
 * SpeechEvaluationDetailPage.tsx
 * 
 * 설명: 발표 분석의 세부 결과 페이지 (말하기 속도 또는 음량 분석)
 * 
 * URL 파라미터:
 * - type: 'speed' 또는 'volume' (분석 유형)
 * 
 * 데이터 흐름:
 * 1. AnalysisDashboardPage에서 사용자가 특정 차트(속도/음량)를 클릭하면 이 페이지로 이동
 * 2. URL 파라미터 'type'에 따라 해당 분석 결과를 표시
 * 3. 백엔드 API에서 분석 데이터를 가져와 차트와 피드백으로 시각화
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
  Slide,
  IconButton,
  Alert,
  CircularProgress,
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
  BarChart,
  Bar,
  Cell,
} from 'recharts';
import { useNavigate, useParams } from 'react-router-dom';
import {
  ArrowBack as ArrowBackIcon,
  GraphicEq as GraphicEqIcon,
  Speed as SpeedIcon,
  Info as InfoIcon,
} from '@mui/icons-material';
import axios from 'axios';

/**
 * 데이터 타입 정의
 * 
 * AnalysisType: 분석 유형 ('speed' 또는 'volume')
 * ChartEventData: 차트 클릭 이벤트 데이터 구조
 * SpeakingSpeedData: 말하기 속도 데이터 구조 (시간 구간별 WPM값)
 * SpeakingEvaluationData: 말하기 속도 평가 데이터 구조 (시간 구간별 피드백)
 * VolumeAnalysisData: 음량 분석 데이터 구조 (시간 구간별 음량값)
 * VolumeEvaluationData: 음량 평가 데이터 구조 (시간 구간별 피드백)
 * AnalysisResult: 전체 분석 결과 데이터 구조
 * ChartDataPoint: 차트 표시용 데이터 포인트 구조
 */
type AnalysisType = 'speed' | 'volume';

// 이벤트 핸들러를 위한 차트 데이터 타입
interface ChartEventData {
  activeLabel?: string;
  activePayload?: Array<{
    payload: ChartDataPoint;
  }>;
  activeCoordinate?: {
    x: number;
    y: number;
  };
  chartX?: number;
  chartY?: number;
}

/**
 * 백엔드 API에서 반환되는 데이터 타입 정의
 * 
 * 주의: 백엔드 개발자는 이 타입 정의에 맞게 API 응답을 구성해야 함
 */
type SpeakingSpeedData = { 
  [timeRange: string]: number; // 각 시간 구간(예: "00:00-01:00")과 WPM값 (예: 120) 
};

type SpeakingEvaluationData = { 
  [timeRange: string]: string; // 각 시간 구간별 피드백 메시지 
};

type VolumeAnalysisData = { 
  [timeRange: string]: number; // 각 시간 구간과 음량(dB)값 
};

type VolumeEvaluationData = { 
  [timeRange: string]: string; // 각 시간 구간별 피드백 메시지 
};

/**
 * 분석 결과 전체 타입
 * 
 * 백엔드 API 응답은 이 형식을 따라야 함
 * 각 필드의 의미:
 * - _id: 분석 결과 고유 ID (MongoDB ID 형식)
 * - user_id: 사용자 ID
 * - filename: 분석한 영상 파일명
 * - speaking_speed: 시간 구간별 말하기 속도 데이터 (WPM)
 * - volume_analysis: 시간 구간별 음량 데이터 (dB)
 * - speaking_evaluation: 시간 구간별 말하기 속도 피드백
 * - volume_evaluation: 시간 구간별 음량 피드백
 * - message: 오류 또는 정보 메시지
 */
type AnalysisResult = {
  _id?: string;
  user_id?: number;
  filename?: string;
  speaking_speed?: SpeakingSpeedData;
  volume_analysis?: VolumeAnalysisData;
  speaking_evaluation?: SpeakingEvaluationData;
  volume_evaluation?: VolumeEvaluationData;
  message?: string;
};

/**
 * 차트에 표시할 데이터 포인트 형식
 * 
 * time: X축에 표시될 시간 (예: "00:00")
 * value: Y축에 표시될 값 (말하기 속도인 경우 WPM, 음량인 경우 dB)
 * timeRange: 원본 데이터의 시간 범위 (예: "00:00-01:00")
 */
interface ChartDataPoint {
  time: string;
  value: number;
  timeRange?: string;
}

const SpeechEvaluationDetailPage: React.FC = () => {
  const navigate = useNavigate();
  const params = useParams();
  // URL 파라미터에서 분석 유형 추출 ('speed' 또는 'volume')
  const type = params.type as AnalysisType;
  
  // 상태 관리
  const [analysisData, setAnalysisData] = useState<AnalysisResult | null>(null);
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
  const [selectedTimeRange, setSelectedTimeRange] = useState<string | null>(null);
  const [selectedFeedback, setSelectedFeedback] = useState<string>('구간을 선택하면 상세 피드백이 표시됩니다.');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // UI 애니메이션 상태
  const [showContent, setShowContent] = useState(false);
  const [showFeedback, setShowFeedback] = useState(false);
  
  /**
   * 분석 타입 유효성 검사
   * 
   * URL 파라미터 'type'이 'speed' 또는 'volume'이 아니면 에러 표시
   */
  useEffect(() => {
    if (type !== 'speed' && type !== 'volume') {
      setError('잘못된 분석 유형입니다.');
      // 잘못된 URL 파라미터인 경우 대시보드로 리디렉션하는 것이 좋음
      // 현재는 alert만 표시
      alert("잘못된 분석 유형");
    }
  }, [type, navigate]);
  
  // 분석 타입에 따른 텍스트 설정
  const analysisTypeTitle = type === 'speed' ? '말하기 속도 분석' : '음량 분석';
  const analysisTypeUnit = type === 'speed' ? 'WPM' : 'dB';
  
  // 페이지 로딩 시 애니메이션 설정
  useEffect(() => {
    const timer1 = setTimeout(() => setShowContent(true), 300);
    return () => clearTimeout(timer1);
  }, []);
  
  /**
   * 백엔드 API에서 분석 데이터 가져오기
   * 
   * 백엔드 개발자 참고:
   * 1. API 엔드포인트: '/api/analysis/{type}' (여기서 type은 'speed' 또는 'volume')
   * 2. 응답 형식은 AnalysisResult 타입을 따라야 함
   * 3. 현재는 개발 테스트를 위한 목업 데이터 사용 중
   * 4. 실제 구현 시 아래 주석된 axios 호출 코드 사용
   */
  useEffect(() => {
    if (type !== 'speed' && type !== 'volume') return;
    
    const fetchAnalysisData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        /**
         * 백엔드 API 연동 시 코드 예제:
         * 
         * const response = await axios.get(`/api/analysis/${type}`);
         * setAnalysisData(response.data);
         * processChartData(response.data);
         */
        
        // 개발 테스트를 위한 예시 데이터
        setTimeout(() => {
          const mockData: AnalysisResult = {
            speaking_speed: {
              "00:00-01:00": 115,
              "01:00-02:00": 125,
              "02:00-03:00": 140,
              "03:00-04:00": 130,
              "04:00-05:00": 121,
              "05:00-06:00": 115,
              "06:00-07:00": 105,
              "07:00-08:00": 128,
            },
            volume_analysis: {
              "00:00-01:00": 65,
              "01:00-02:00": 70,
              "02:00-03:00": 72,
              "03:00-04:00": 65,
              "04:00-05:00": 60,
              "05:00-06:00": 68,
              "06:00-07:00": 72,
              "07:00-08:00": 75,
            },
            speaking_evaluation: {
              "00:00-01:00": "이 구간의 속도는 듣기에 적당합니다.",
              "01:00-02:00": "이 구간의 속도는 듣기에 적당합니다.",
              "02:00-03:00": "이 구간은 말하는 속도가 빨라 이해하기 어려울 수 있습니다.",
              "03:00-04:00": "이 구간의 속도는 듣기에 적당합니다.",
              "04:00-05:00": "이 구간의 속도는 듣기에 적당합니다.",
              "05:00-06:00": "이 구간의 속도는 듣기에 적당합니다.",
              "06:00-07:00": "이 구간은 말하는 속도가 느려 지루하게 느껴질 수 있습니다.",
              "07:00-08:00": "이 구간의 속도는 듣기에 적당합니다.",
            },
            volume_evaluation: {
              "00:00-01:00": "이 구간의 음량이 적절하여 메시지 전달력이 높았습니다.",
              "01:00-02:00": "이 구간의 음량이 적절하여 메시지 전달력이 높았습니다.",
              "02:00-03:00": "이 구간의 음량이 적절하여 메시지 전달력이 높았습니다.",
              "03:00-04:00": "이 구간의 음량이 적절하여 메시지 전달력이 높았습니다.",
              "04:00-05:00": "이 구간의 음량이 작아 강조가 부족했습니다.",
              "05:00-06:00": "이 구간의 음량이 적절하여 메시지 전달력이 높았습니다.",
              "06:00-07:00": "이 구간의 음량이 적절하여 메시지 전달력이 높았습니다.",
              "07:00-08:00": "이 구간의 음량이 너무 크게 들려 다소 공격적으로 느껴질 수 있습니다.",
            }
          };
          
          setAnalysisData(mockData);
          processChartData(mockData);
        }, 1000);
      } catch (err) {
        if (axios.isAxiosError(err) && err.response) {
          setError(`데이터를 불러오는 중 오류가 발생했습니다: ${err.response.status}`);
        } else {
          setError('데이터를 불러오는 중 오류가 발생했습니다.');
        }
      } finally {
        setLoading(false);
      }
    };
    
    fetchAnalysisData();
  }, [type, navigate]);
  
  /**
   * 백엔드 데이터를 차트 형식으로 변환하는 함수
   * 
   * API에서 받은 데이터를 차트에 표시할 수 있는 형식으로 변환
   * 
   * @param data AnalysisResult 타입의 백엔드 응답 데이터
   */
  const processChartData = (data: AnalysisResult) => {
    if (!data) return;
    
    // 분석 타입에 따라 적절한 데이터 선택 (말하기 속도 또는 음량)
    const sourceData = type === 'speed' ? data.speaking_speed : data.volume_analysis;
    if (!sourceData) return;
    
    // 데이터 형식 변환: { "00:00-01:00": 120 } -> { time: "00:00", value: 120, timeRange: "00:00-01:00" }
    const formattedData: ChartDataPoint[] = Object.entries(sourceData).map(([timeRange, value]) => ({
      time: timeRange.split('-')[0], // 시작 시간만 X축에 표시
      value: value,
      timeRange: timeRange // 전체 타임레인지 저장 (피드백 표시 시 필요)
    }));
    
    setChartData(formattedData);
  };
  
  /**
   * 차트 바 클릭 이벤트 핸들러
   * 
   * 사용자가 차트의 특정 구간을 클릭하면 해당 구간의 피드백을 표시
   * 
   * @param data 차트 클릭 이벤트 데이터
   */
  const handleChartClick = (data: ChartEventData) => {
    if (!analysisData || !data.activePayload || data.activePayload.length === 0) return;
    
    const payload = data.activePayload[0].payload;
    const selectedRange = payload.timeRange;
    
    if (selectedRange) {
      setSelectedTimeRange(selectedRange);
      
      // 분석 타입에 따라 적절한 피드백 설정
      if (type === 'speed' && analysisData.speaking_evaluation) {
        setSelectedFeedback(analysisData.speaking_evaluation[selectedRange] || '이 구간에 대한 피드백이 없습니다.');
      } else if (type === 'volume' && analysisData.volume_evaluation) {
        setSelectedFeedback(analysisData.volume_evaluation[selectedRange] || '이 구간에 대한 피드백이 없습니다.');
      }
      
      // 피드백 영역 갱신 애니메이션
      setShowFeedback(false);
      setTimeout(() => {
        setShowFeedback(true);
      }, 300);
    }
  };
  
  /**
   * 차트 바의 색상 결정 함수
   * 
   * 값에 따라 다른 색상 반환:
   * - 말하기 속도(WPM): 너무 느리거나(110 미만) 너무 빠르면(130 초과) 주황색, 적정 속도면 녹색
   * - 음량(dB): 너무 작거나(65 미만) 너무 크면(70 초과) 주황색, 적정 음량이면 녹색
   * 
   * @param value 차트에 표시될 값 (WPM 또는 dB)
   * @returns 색상 코드 (HEX)
   */
  const getBarColor = (value: number): string => {
    if (type === 'speed') {
      // 속도(WPM) 차트 색상
      if (value < 110) return '#ff8c00'; // 너무 느림
      if (value > 130) return '#ff8c00'; // 너무 빠름
      return '#4caf50'; // 적정 속도
    } else {
      // 음량(dB) 차트 색상
      if (value < 65) return '#ff8c00'; // 음량이 작음
      if (value > 70) return '#ff8c00'; // 음량이 큼
      return '#4caf50'; // 적정 음량
    }
  };
  
  /**
   * 차트 참조선 설정 함수
   * 
   * 분석 타입에 따라 적절한 참조선 반환:
   * - 말하기 속도: 110 WPM과 130 WPM에 참조선 (적정 속도 범위)
   * - 음량: 65 dB과 70 dB에 참조선 (적정 음량 범위)
   * 
   * @returns 참조선 JSX 요소
   */
  const getReferenceLines = () => {
    if (type === 'speed') {
      return (
        <>
          <ReferenceLine y={110} stroke="#999" strokeDasharray="3 3" />
          <ReferenceLine y={130} stroke="#999" strokeDasharray="3 3" />
        </>
      );
    } else {
      return (
        <>
          <ReferenceLine y={65} stroke="#999" strokeDasharray="3 3" />
          <ReferenceLine y={70} stroke="#999" strokeDasharray="3 3" />
        </>
      );
    }
  };
  
  /**
   * 대시보드로 돌아가는 함수
   * 
   * 사용자가 뒤로 가기 버튼을 클릭하면 분석 대시보드로 이동
   */
  const handleNavigateBack = () => {
    navigate('/analysis');
  };

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
        {/* 헤더 섹션 */}
        <Box sx={{ mb: 5, display: 'flex', alignItems: 'center' }}>
          <IconButton 
            aria-label="돌아가기" 
            onClick={handleNavigateBack}
            sx={{ mr: 2 }}
          >
            <ArrowBackIcon />
          </IconButton>
          <Box>
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
              {analysisTypeTitle}
            </Typography>
            <Typography 
              variant="body1" 
              color="text.secondary"
              sx={{ mt: 3 }}
            >
              구간별 {type === 'speed' ? '말하기 속도' : '음량'} 데이터와 맞춤형 피드백을 확인하세요.
            </Typography>
          </Box>
        </Box>
        
        {/* 에러 메시지 표시 */}
        {error && (
          <Alert 
            severity="error" 
            sx={{ mb: 3 }}
            onClose={() => setError(null)}
          >
            {error}
          </Alert>
        )}
        
        <Fade in={showContent} timeout={1000}>
          <Grid container spacing={4}>
            {/* 왼쪽 컬럼: 차트 */}
            <Grid item xs={12} md={7}>
              <Slide direction="right" in={showContent} timeout={800}>
                <Paper 
                  elevation={0}
                  sx={{ 
                    p: 3, 
                    borderRadius: 3,
                    boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
                    height: '100%',
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
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="h6" component="h2" fontWeight={600}>
                      구간별 {type === 'speed' ? '말하기 속도' : '음량'} ({analysisTypeUnit})
                    </Typography>
                    {type === 'speed' ? <SpeedIcon /> : <GraphicEqIcon />}
                  </Box>
                  
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                    {type === 'speed' 
                      ? '구간을 클릭하여 특정 구간의 말하기 속도에 대한 피드백을 확인하세요.' 
                      : '구간을 클릭하여 특정 구간의 음량에 대한 피드백을 확인하세요.'}
                  </Typography>
                  
                  {loading ? (
                    <Box sx={{ height: 300, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <CircularProgress size={40} />
                    </Box>
                  ) : (
                    <Box sx={{ height: 300, width: '100%' }}>
                      <ResponsiveContainer width="100%" height="100%">
                        {type === 'speed' ? (
                          // 말하기 속도 차트 (막대 그래프)
                          <BarChart 
                            data={chartData} 
                            margin={{ top: 10, right: 10, left: 10, bottom: 10 }}
                            onClick={handleChartClick}
                          >
                            <CartesianGrid strokeDasharray="3 3" vertical={false} />
                            <XAxis dataKey="time" />
                            <YAxis domain={[90, 150]} />
                            <Tooltip 
                              formatter={(value) => [`${value} WPM`, '말하기 속도']}
                              contentStyle={{ 
                                backgroundColor: '#fff', 
                                borderRadius: '8px',
                                boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                                border: 'none' 
                              }} 
                            />
                            {getReferenceLines()}
                            <Bar 
                              dataKey="value" 
                              cursor="pointer"
                            >
                              {chartData.map((entry, index) => (
                                <Cell 
                                  key={`cell-${index}`} 
                                  fill={getBarColor(entry.value)}
                                  opacity={selectedTimeRange === entry.timeRange ? 1 : 0.7}
                                  stroke={selectedTimeRange === entry.timeRange ? '#000' : 'none'}
                                  strokeWidth={1}
                                />
                              ))}
                            </Bar>
                          </BarChart>
                        ) : (
                          // 음량 차트 (라인 그래프)
                          <LineChart 
                            data={chartData} 
                            margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
                            onClick={handleChartClick}
                          >
                            <CartesianGrid strokeDasharray="3 3" vertical={false} />
                            <XAxis dataKey="time" />
                            <YAxis domain={[50, 85]} />
                            <Tooltip 
                              formatter={(value) => [`${value} dB`, '음량']}
                              contentStyle={{ 
                                backgroundColor: '#fff', 
                                borderRadius: '8px',
                                boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                                border: 'none' 
                              }} 
                            />
                            {getReferenceLines()}
                            <Line 
                              type="monotone" 
                              dataKey="value" 
                              stroke="#000" 
                              strokeWidth={3} 
                              dot={{ r: 4 }}
                              activeDot={{ 
                                r: 6, 
                                strokeWidth: 1, 
                                stroke: '#FFF',
                              }}
                            />
                          </LineChart>
                        )}
                      </ResponsiveContainer>
                    </Box>
                  )}
                  
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 3 }}>
                    <Typography variant="caption" color="text.secondary">
                      {type === 'speed' ? '권장 말하기 속도: 120-150 WPM' : '권장 음량 범위: 65-70 dB'}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      그래프의 구간을 클릭하여 상세 정보 확인
                    </Typography>
                  </Box>
                </Paper>
              </Slide>
            </Grid>
            
            {/* 오른쪽 컬럼: 피드백 */}
            <Grid item xs={12} md={5}>
              <Slide direction="left" in={showContent} timeout={800}>
                <Card 
                  elevation={0}
                  sx={{ 
                    borderRadius: 3,
                    boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
                    height: '100%',
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
                  <CardContent sx={{ p: 4, flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
                    <Typography 
                      variant="h5" 
                      component="h2" 
                      fontWeight={600}
                      sx={{ mb: 3 }}
                    >
                      상세 피드백
                    </Typography>
                    
                    <Box 
                      sx={{ 
                        p: 3, 
                        borderRadius: 3, 
                        bgcolor: '#f5f5f5',
                        mb: 3,
                        display: 'flex',
                        alignItems: 'center'
                      }}
                    >
                      <InfoIcon sx={{ mr: 2, color: '#666' }} />
                      <Typography variant="body2" color="text.secondary">
                        {type === 'speed' 
                          ? '말하기 속도는 효과적인 메시지 전달을 위해 중요합니다. 110-130 WPM이 일반적으로 권장됩니다.' 
                          : '음량은 청중의 주의를 끌고 메시지의 중요성을 강조하는 데 중요합니다. 65-70 dB이 일반적으로 권장됩니다.'}
                      </Typography>
                    </Box>
                    
                    <Divider sx={{ mb: 3 }} />
                    
                    <Box>
                      <Typography 
                        variant="subtitle1" 
                        fontWeight={600}
                        sx={{ mb: 2 }}
                      >
                        {selectedTimeRange ? `선택 구간: ${selectedTimeRange}` : '구간을 선택하세요'}
                      </Typography>
                      
                      {/* 피드백 메시지 표시 영역 - 클릭한 구간의 피드백 메시지 표시 */}
                      <Fade in={showFeedback} timeout={500}>
                        <Paper 
                          elevation={0}
                          sx={{ 
                            p: 3, 
                            borderRadius: 3,
                            bgcolor: '#fff',
                            border: '1px solid #eaeaea',
                            boxShadow: '0 2px 10px rgba(0,0,0,0.05)',
                            mb: 3
                          }}
                        >
                          <Typography variant="body1">
                            {selectedFeedback}
                          </Typography>
                        </Paper>
                      </Fade>
                    </Box>
                    
                    {/* 개선 팁 영역 - 분석 유형에 따른 일반적인 개선 팁 표시 */}
                    <Box sx={{ mt: 'auto' }}>
                      <Typography variant="h6" fontWeight={600} sx={{ mb: 2 }}>
                        개선 팁
                      </Typography>
                      
                      <Box 
                        sx={{ 
                          p: 3, 
                          borderRadius: 3, 
                          bgcolor: '#f8f8f8',
                          border: '1px solid #eaeaea',
                        }}
                      >
                        {type === 'speed' ? (
                          <ul style={{ paddingLeft: '1.5rem', margin: 0 }}>
                            <li>
                              <Typography variant="body2" sx={{ mb: 1 }}>
                                중요한 내용에서는 속도를 줄여 강조하세요.
                              </Typography>
                            </li>
                            <li>
                              <Typography variant="body2" sx={{ mb: 1 }}>
                                문장 사이에 잠시 멈추는 것이 청중이 내용을 이해하는 데 도움이 됩니다.
                              </Typography>
                            </li>
                            <li>
                              <Typography variant="body2">
                                연습 중에 말하기 속도를 체크하며 일정한 속도를 유지하는 훈련을 하세요.
                              </Typography>
                            </li>
                          </ul>
                        ) : (
                          <ul style={{ paddingLeft: '1.5rem', margin: 0 }}>
                            <li>
                              <Typography variant="body2" sx={{ mb: 1 }}>
                                중요한 내용에서는 음량을 약간 높여 강조할 수 있습니다.
                              </Typography>
                            </li>
                            <li>
                              <Typography variant="body2" sx={{ mb: 1 }}>
                                일관된 음량을 유지하는 것이 전문성을 높이는 데 도움이 됩니다.
                              </Typography>
                            </li>
                            <li>
                              <Typography variant="body2">
                                너무 큰 소리나 작은 소리는 청중의 집중력을 떨어뜨릴 수 있습니다.
                              </Typography>
                            </li>
                          </ul>
                        )}
                      </Box>
                    </Box>
                  </CardContent>
                </Card>
              </Slide>
            </Grid>
          </Grid>
        </Fade>
        
        {/* 하단 탐색 버튼 영역 */}
        <Box 
          sx={{ 
            display: 'flex', 
            justifyContent: 'space-between',
            mt: 4 
          }}
        >
          <Button
            variant="outlined"
            startIcon={<ArrowBackIcon />}
            onClick={handleNavigateBack}
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
            대시보드로 돌아가기
          </Button>
          
          {/* 다른 분석 유형으로 이동하는 버튼 */}
          <Button
            variant="outlined"
            onClick={() => navigate(type === 'speed' ? '/analysis/volume' : '/analysis/speed')}
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
            {type === 'speed' ? '음량 분석' : '말하기 속도 분석'} 보기
          </Button>
        </Box>
        
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
            © 2025 PRESENT INSIGHT. All rights reserved.
          </Typography>
        </Box>
      </Container>
    </Box>
  );
};

export default SpeechEvaluationDetailPage;