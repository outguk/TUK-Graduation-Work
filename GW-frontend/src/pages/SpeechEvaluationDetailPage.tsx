// /**
//  * SpeechEvaluationDetailPage.tsx
//  * 
//  * 설명: 발표 분석의 세부 결과 페이지 (말하기 속도 또는 음량 분석)
//  * 
//  * URL 파라미터:
//  * - type: 'speed' 또는 'volume' (분석 유형)
//  * - id: 발표 ID (optional, 없으면 가장 최근 발표 분석 표시)
//  * 
//  * 데이터 흐름:
//  * 1. AnalysisDashboardPage에서 사용자가 특정 차트(속도/음량)를 클릭하면 이 페이지로 이동
//  * 2. URL 파라미터 'type'과 'id'에 따라 해당 분석 결과를 표시
//  * 3. mockAnalysisData에서 분석 데이터를 가져와 차트와 피드백으로 시각화
//  */

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
  ReferenceArea
} from 'recharts';
import { useNavigate, useParams } from 'react-router-dom';
import {
  ArrowBack as ArrowBackIcon,
  GraphicEq as GraphicEqIcon,
  Speed as SpeedIcon,
  Info as InfoIcon,
} from '@mui/icons-material';
import axios from 'axios';

interface ChartDataPoint {
  time: string;
  wpm?: number;
  db?: number;
  timeRange: string;
}

interface SegmentEvaluation {
  start?: number;
  time_stamps?: [number, number];
  feedback: string;
}

interface PresentationAnalysis {
  _id: string;
  filename: string;
  speaking_speed?: { segment_wpm: Array<{ start: number; end: number; wpm: number }> };
  volume_analysis?: { segment_data: Array<{ time_stamps: [number, number]; db: number }> };
  speaking_evaluation?: { segment_evaluations: SegmentEvaluation[] };
  volume_evaluation?: { segment_evaluations: SegmentEvaluation[] };
  title: string;
}

interface ChartEventData {
  activeLabel?: string;
  activePayload?: Array<{
    payload: ChartDataPoint;
  }>;
  activeCoordinate?: {
    x: number;
    y: number;
  }
  chartX?: number;
  chartY?: number;
}

const SpeechEvaluationDetailPage: React.FC = () => {
  const navigate = useNavigate();
  const { type, presentationId } = useParams<{ type: string; presentationId: string }>();

  const [presentation, setPresentation] = useState<PresentationAnalysis | null>(null);
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);
  const [selectedTimeRange, setSelectedTimeRange] = useState<string | null>(null);
  const [selectedFeedback, setSelectedFeedback] = useState<string>('구간을 선택하면 상세 피드백이 표시됩니다.');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showContent, setShowContent] = useState(false);
  const [showFeedback, setShowFeedback] = useState(false);

  useEffect(() => {
    if (!type) return;
    if (type !== 'speed' && type !== 'volume') {
      setError('잘못된 분석 유형입니다.');
    }
  }, [type]);

  const analysisTypeTitle = type === 'speed' ? '말하기 속도 분석' : '음량 분석';
  const analysisTypeUnit = type === 'speed' ? 'WPM' : 'dB';

  useEffect(() => {
    const timer1 = setTimeout(() => setShowContent(true), 300);
    return () => clearTimeout(timer1);
  }, []);

  useEffect(() => {
    if (type !== 'speed' && type !== 'volume') return;

    const fetchPresentationData = async () => {
      setLoading(true);
      setError(null);

      try {
        const token = localStorage.getItem('token');
        if (!token) {
          throw new Error('인증 토큰이 없습니다.');
        }

        let response;
        if (presentationId) {
          response = await axios.get(`/spring/api/get-analysis`, {
            params: { filename: presentationId },
            headers: { Authorization: `Bearer ${token}` }
          });
        } else {
          response = await axios.get(`/spring/api/my-analyses`, {
            headers: { Authorization: `Bearer ${token}` }
          });
          if (response.data.analyses && response.data.analyses.length > 0) {
            response.data = response.data.analyses[0];
          } else {
            throw new Error('분석 데이터가 없습니다.');
          }
        }

        const data = response.data;
        const presentationData: PresentationAnalysis = {
          _id: data._id,
          filename: data.filename,
          speaking_speed: data.speaking_speed,
          volume_analysis: data.volume_analysis,
          speaking_evaluation: data.speaking_evaluation,
          volume_evaluation: data.volume_evaluation,
          title: data.filename.split('.')[0],
        };

        setPresentation(presentationData);
        processChartData(presentationData);
      } catch (err: any) {
        console.error("발표 데이터 로드 에러:", err);
        setError(err.response?.data?.error || '데이터를 불러오는 중 오류가 발생했습니다.');
      } finally {
        setLoading(false);
      }
    };

    fetchPresentationData();
  }, [type, presentationId]);

  const processChartData = (data: PresentationAnalysis) => {
    if (!data) return;

    const formattedData: ChartDataPoint[] = [];

    if (type === 'speed' && data.speaking_speed) {
      data.speaking_speed.segment_wpm.forEach((segment) => {
        const timeRangeStr = `${formatTime(segment.start)}-${formatTime(segment.end)}`;
        formattedData.push({
          time: formatTime(segment.start),
          wpm: segment.wpm,
          timeRange: timeRangeStr
        });
      });
    } else if (type === 'volume' && data.volume_analysis) {
      data.volume_analysis.segment_data.forEach((segment) => {
        const timeRangeStr = `${formatTime(segment.time_stamps[0])}-${formatTime(segment.time_stamps[1])}`;
        formattedData.push({
          time: formatTime(segment.time_stamps[0]),
          db: segment.db,
          timeRange: timeRangeStr
        });
      });
    }

    setChartData(formattedData);

    if (formattedData.length > 0 && formattedData[0].timeRange) {
      handleTimeRangeSelect(formattedData[0].timeRange);
    }
  };

  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  const handleTimeRangeSelect = (timeRange: string) => {
    if (!presentation) return;

    setSelectedTimeRange(timeRange);

    const startTimeStr = timeRange.split('-')[0];
    let feedback = '이 구간에 대한 피드백이 없습니다.';

    if (type === 'speed' && presentation.speaking_evaluation) {
      for (const segment of presentation.speaking_evaluation.segment_evaluations) {
        if (formatTime(segment.start!) === startTimeStr) {
          feedback = segment.feedback;
          break;
        }
      }
    } else if (type === 'volume' && presentation.volume_evaluation) {
      for (const segment of presentation.volume_evaluation.segment_evaluations) {
        if (formatTime(segment.time_stamps![0]) === startTimeStr) {
          feedback = segment.feedback;
          break;
        }
      }
    }

    setSelectedFeedback(feedback);

    setShowFeedback(false);
    setTimeout(() => {
      setShowFeedback(true);
    }, 300);
  };

  const handleChartClick = (data: ChartEventData) => {
    if (!data.activePayload || data.activePayload.length === 0) return;

    const payload = data.activePayload[0].payload;
    const selectedRange = payload.timeRange;

    if (selectedRange) {
      handleTimeRangeSelect(selectedRange);
    }
  };

// getBarColor와 getReferenceLines 함수 업데이트
// 정상 범위를 60-70dB로 설정하고 색상 체계 추가

// 정상 범위와 심각한 벗어남의 기준을 정의
const SPEED_NORMAL_MIN = 100;
const SPEED_NORMAL_MAX = 140;
const SPEED_SEVERE_DEVIATION = 30; // 정상 범위에서 30 WPM 이상 벗어나면 심각하게 간주

const VOLUME_NORMAL_MIN = 60;
const VOLUME_NORMAL_MAX = 70;
const VOLUME_SEVERE_DEVIATION = 15; // 정상 범위에서 10 dB 이상 벗어나면 심각하게 간주

// 음량에 따른 색상을 결정하는 함수
const getVolumeColor = (db: number | undefined | null): string => {
  if (db === undefined || db === null) return '#888888'; // 값이 없는 경우
  
  // 정상 범위 내
  if (db >= VOLUME_NORMAL_MIN && db <= VOLUME_NORMAL_MAX) {
    return '#4caf50'; // 정상 - 초록색
  }
  
  // 심각하게 벗어난 경우
  if (db < VOLUME_NORMAL_MIN - VOLUME_SEVERE_DEVIATION || db > VOLUME_NORMAL_MAX + VOLUME_SEVERE_DEVIATION) {
    return '#e53935'; // 심각한 벗어남 - 빨간색
  }
  
  // 약간 벗어난 경우
  return '#ff8c00'; // 약간 벗어남 - 주황색
};

// 말하기 속도에 따른 색상을 결정하는 함수
const getSpeedColor = (wpm: number | undefined | null): string => {
  if (wpm === undefined || wpm === null) return '#888888'; // 값이 없는 경우
  
  // 정상 범위 내
  if (wpm >= SPEED_NORMAL_MIN && wpm <= SPEED_NORMAL_MAX) {
    return '#4caf50'; // 정상 - 초록색
  }
  
  // 심각하게 벗어난 경우
  if (wpm < SPEED_NORMAL_MIN - SPEED_SEVERE_DEVIATION || wpm > SPEED_NORMAL_MAX + SPEED_SEVERE_DEVIATION) {
    return '#e53935'; // 심각한 벗어남 - 빨간색
  }
  
  // 약간 벗어난 경우
  return '#ff8c00'; // 약간 벗어남 - 주황색
};

const getReferenceLines = () => {
  if (type === 'speed') {
    return (
      <>
        <ReferenceLine y={SPEED_NORMAL_MIN} stroke="#999" strokeDasharray="3 3" />
        <ReferenceLine y={SPEED_NORMAL_MAX} stroke="#999" strokeDasharray="3 3" />
      </>
    );
  } else {
    return (
      <>
        <ReferenceArea y1={VOLUME_NORMAL_MIN} y2={VOLUME_NORMAL_MAX} fill="#e6f4ea" fillOpacity={0.5} />
        <ReferenceLine y={VOLUME_NORMAL_MIN} stroke="#999" strokeDasharray="3 3" />
        <ReferenceLine y={VOLUME_NORMAL_MAX} stroke="#999" strokeDasharray="3 3" />
      </>
    );
  }
};

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
              {presentation && ` - ${presentation.title}`}
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
                          <BarChart 
                            data={chartData} 
                            margin={{ top: 10, right: 10, left: 10, bottom: 10 }}
                            onClick={handleChartClick}
                          >
                            <CartesianGrid strokeDasharray="3 3" vertical={false} />
                            <XAxis dataKey="time" />
                            <YAxis domain={[80, 200]} />
                            
                            {/* 정상 범위 영역 표시 */}
                            <ReferenceArea y1={SPEED_NORMAL_MIN} y2={SPEED_NORMAL_MAX} fill="#e6f4ea" fillOpacity={0.5} />
                            
                            <Tooltip 
                              formatter={(wpm) => [`${wpm} WPM`, '말하기 속도']}
                              contentStyle={{ 
                                backgroundColor: '#fff', 
                                borderRadius: '8px',
                                boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                                border: 'none' 
                              }} 
                            />
                            <ReferenceLine y={SPEED_NORMAL_MIN} stroke="#999" strokeDasharray="3 3" />
                            <ReferenceLine y={SPEED_NORMAL_MAX} stroke="#999" strokeDasharray="3 3" />
                            <Bar 
                              dataKey="wpm" 
                              cursor="pointer"
                            >
                              {chartData.map((entry, index) => (
                                <Cell 
                                  key={`cell-${index}`} 
                                  fill={getSpeedColor(entry.wpm)}
                                  opacity={selectedTimeRange === entry.timeRange ? 1 : 0.7}
                                  stroke={selectedTimeRange === entry.timeRange ? '#000' : 'none'}
                                  strokeWidth={1}
                                />
                              ))}
                            </Bar>
                          </BarChart>
                        ) : (
                          // LineChart 코드
                          <LineChart 
                            data={chartData} 
                            margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
                            onClick={handleChartClick}
                          >
                            <CartesianGrid strokeDasharray="3 3" vertical={false} />
                            <XAxis dataKey="time" />
                            <YAxis domain={[30, 90]} />
                            
                            <Tooltip 
                              formatter={(db) => [`${db} dB`, '음량']}
                              contentStyle={{ 
                                backgroundColor: '#fff', 
                                borderRadius: '8px',
                                boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                                border: 'none' 
                              }} 
                            />
                            
                            {/* 기존 함수 사용 */}
                            {getReferenceLines()}
                            
                            <Line 
                              type="monotone" 
                              dataKey="db" 
                              stroke="#000" 
                              strokeWidth={3} 
                              dot={(props) => {
                                const { cx, cy, payload } = props;
                                const db = payload.db;
                                
                                // db 값이 없는 경우 처리
                                if (db === undefined || db === null) {
                                  return <circle cx={cx} cy={cy} r={0} fill="none" />;
                                }
                                
                                // 음량에 따른 색상 결정
                                const fillColor = getVolumeColor(db);
                                
                                // 선택된 점 강조
                                const isSelected = selectedTimeRange === payload.timeRange;
                                const size = isSelected ? 6 : 4;
                                
                                return (
                                  <circle
                                    cx={cx}
                                    cy={cy}
                                    r={size}
                                    fill={fillColor}
                                    stroke="#FFF"
                                    strokeWidth={1}
                                    style={{ cursor: 'pointer' }}
                                  />
                                );
                              }}
                              activeDot={{ 
                                r: 6, 
                                strokeWidth: 1, 
                                stroke: '#000000',
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
          
          <Button
            variant="outlined"
            onClick={() => {
              const newType = type === 'speed' ? 'volume' : 'speed';
              navigate(presentationId 
                ? `/analysis/${newType}/${presentationId}` 
                : `/analysis/${newType}`
              );
            }}
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