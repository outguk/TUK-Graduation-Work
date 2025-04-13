/**
 * NonverbalEvaluationDetailPage.tsx
 * 
 * 설명: 발표 분석의 비언어적 요소 세부 결과 페이지
 * 
 * 기능:
 * 1. 타임라인 기반 비디오 세그먼트 표시 및 상호작용
 * 2. 비언어적 행동 카테고리별 발생 빈도 차트
 * 3. 선택된 구간의 비언어적 행동 피드백 표시
 */

import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Box,
  Container,
  Typography,
  Grid,
  Paper,
  Button,
  Fade,
  Slide,
  IconButton,
  Alert,
  CircularProgress,
  Divider,
} from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { useNavigate } from 'react-router-dom';
import {
  ArrowBack as ArrowBackIcon,
  Info as InfoIcon,
  VideoLibrary as VideoLibraryIcon,
  Category as CategoryIcon,
  ChevronLeft as ChevronLeftIcon,
  ChevronRight as ChevronRightIcon,
} from '@mui/icons-material';
import axios from 'axios';

/**
 * 데이터 타입 정의
 */
// 분석된 비언어적 행동 클래스 (예: "자세(비스듬히)")
interface BehaviorClass {
  class: string;
  probability: number;
}

// 비언어적 분석 결과의 시간 구간 단위 - 실제 API 응답 구조에 맞게 수정
interface NonverbalSegment {
  sample_number: number;
  time_range: string;
  frame_range: string;
  frame_dir: string;
  is_normal: boolean;
  wrist_distance: number;
  top_classes: BehaviorClass[];
}

// 비언어적 분석 데이터 타입 - 배열로 수정
type NonverbalAnalysisData = NonverbalSegment[];

// 비언어적 분석 API 응답 타입
interface NonverbalAnalysisResult {
  nonverbal_analysis: NonverbalAnalysisData;
  message?: string;
}

// 행동 카테고리 발생 빈도 타입
interface CategoryCount {
  category: string;
  count: number;
}

// 행동 카테고리 정의 및 색상 매핑
const BEHAVIOR_CATEGORIES: { [key: string]: string } = {
  '자세': '#63e6be',
  '손동작': '#ff6b6b',
  '머리동작': '#4dabf7',
  '팔동작': '#4dabf7'
};

// 행동별 피드백 매핑 (행동 -> 개선 팁)
const BEHAVIOR_FEEDBACK: { [key: string]: string } = {
  "손동작(머리)": "머리를 만지는 습관은 불안한 인상을 줄 수 있어요. 손은 안정된 위치에 두는 것이 좋아요.",
  "손동작(얼굴)": "얼굴을 만지는 행동은 산만해 보일 수 있어요. 청중과의 시선 유지에 집중해보세요.",
  "손동작(몸긁기)": "몸을 긁는 동작은 불편함을 드러낼 수 있어요. 손은 가볍게 모으거나 제스처로 활용해보세요.",
  "손동작(손톱)": "손톱을 만지거나 뜯는 동작은 긴장감을 전달해요. 손을 자연스럽게 두는 연습이 필요해요.",
  "머리동작(고개흔들기)": "고개를 자주 흔드는 것은 산만한 인상을 줄 수 있어요. 시선과 자세를 고정해보세요.",
  "머리동작(좌우흔들기)": "머리를 좌우로 흔드는 습관은 주의를 흩뜨릴 수 있어요. 차분한 고개 움직임을 유지하세요.",
  "머리동작(숙이기)": "고개를 숙이는 자세는 자신감 부족으로 보일 수 있어요. 시선을 정면으로 유지해보세요.",
  "팔동작(뒷짐)": "뒷짐은 청중과의 거리감을 줄 수 있어요. 앞에 두고 자연스러운 제스처를 사용하는 것이 좋습니다.",
  "팔동작(무의미반동)": "불필요한 팔의 움직임은 발표 흐름을 방해할 수 있어요. 의미 있는 제스처만 사용하는 연습이 필요해요.",
  "자세(좌우흔들기)": "몸을 좌우로 흔드는 습관은 발표자의 긴장을 드러냅니다. 중심을 잡고 안정된 자세를 유지해보세요.",
  "자세(비스듬히)": "비스듬한 자세는 집중이 부족해 보일 수 있어요. 정면을 향한 단단한 자세가 신뢰감을 줍니다.",
  "자세(비비꼬기)": "다리를 꼬는 자세는 불안정하고 산만한 인상을 줄 수 있어요. 두 다리를 안정적으로 두는 자세를 연습하세요."
};

const NonverbalEvaluationDetailPage: React.FC = () => {
  const navigate = useNavigate();
  const timelineRef = useRef<HTMLDivElement>(null);
  
  // 상태 관리
  const [analysisData, setAnalysisData] = useState<NonverbalAnalysisData | null>(null);
  const [categoryCountData, setCategoryCountData] = useState<CategoryCount[]>([]);
  const [selectedSegmentIndex, setSelectedSegmentIndex] = useState<number | null>(null);
  const [selectedBehavior, setSelectedBehavior] = useState<string>("");
  const [selectedFeedback, setSelectedFeedback] = useState<string>("구간을 선택하면 상세 피드백이 표시됩니다.");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // UI 애니메이션 상태
  const [showContent, setShowContent] = useState(false);
  const [showFeedback, setShowFeedback] = useState(false);
  
  // 페이지 로딩 시 애니메이션 설정
  useEffect(() => {
    const timer = setTimeout(() => setShowContent(true), 300);
    return () => clearTimeout(timer);
  }, []);
  
  /**
   * 타임라인 좌우 스크롤 함수
   */
  const scrollTimeline = (direction: 'left' | 'right') => {
    if (timelineRef.current) {
      const scrollAmount = 200; // 스크롤 양 (픽셀)
      const currentScroll = timelineRef.current.scrollLeft;
      
      timelineRef.current.scrollTo({
        left: direction === 'left' ? currentScroll - scrollAmount : currentScroll + scrollAmount,
        behavior: 'smooth'
      });
    }
  };

  /**
   * 백엔드 API에서 비언어적 분석 데이터 가져오기
   * 
   * API 엔드포인트: /api/analysis/nonverbal
   * 응답 형식: NonverbalAnalysisResult 타입 (nonverbal_analysis 필드 포함)
   */
  useEffect(() => {
    const fetchNonverbalData = async () => {
      setLoading(true);
      setError(null);
      
      try {
        // 실제 API 호출 코드 (주석 해제하여 사용)
        // const response = await axios.get<NonverbalAnalysisResult>('/api/analysis/nonverbal');
        // setAnalysisData(response.data.nonverbal_analysis);
        
        // 개발 테스트를 위한 예시 데이터 - 실제 API 구조에 맞게 수정
        setTimeout(() => {
          const mockData: NonverbalAnalysisResult = {
            nonverbal_analysis: [
              {
                sample_number: 1,
                time_range: "0.00s - 1.00s",
                frame_range: "0 - 9",
                frame_dir: "frame_0",
                is_normal: false,
                wrist_distance: 133.73,
                top_classes: [
                  { class: "자세(비스듬히)", probability: 100 },
                  { class: "자세(고개돌림)", probability: 0 },
                  { class: "손동작(얼굴)", probability: 0 }
                ]
              },
              {
                sample_number: 2,
                time_range: "1.00s - 2.00s",
                frame_range: "10 - 19",
                frame_dir: "frame_1",
                is_normal: false,
                wrist_distance: 145.21,
                top_classes: [
                  { class: "손동작(머리)", probability: 85 },
                  { class: "자세(비스듬히)", probability: 10 },
                  { class: "머리동작(숙이기)", probability: 5 }
                ]
              },
              {
                sample_number: 3,
                time_range: "2.00s - 3.00s",
                frame_range: "20 - 29",
                frame_dir: "frame_2",
                is_normal: true,
                wrist_distance: 120.54,
                top_classes: [
                  { class: "정상", probability: 92 }
                ]
              },
              {
                sample_number: 4,
                time_range: "3.00s - 4.00s",
                frame_range: "30 - 39",
                frame_dir: "frame_3",
                is_normal: false,
                wrist_distance: 139.87,
                top_classes: [
                  { class: "머리동작(좌우흔들기)", probability: 78 },
                  { class: "손동작(얼굴)", probability: 15 },
                  { class: "자세(비비꼬기)", probability: 7 }
                ]
              },
              {
                sample_number: 5,
                time_range: "4.00s - 5.00s",
                frame_range: "40 - 49",
                frame_dir: "frame_4",
                is_normal: true,
                wrist_distance: 125.32,
                top_classes: [
                  { class: "정상", probability: 94 }
                ]
              },
              {
                sample_number: 6,
                time_range: "5.00s - 6.00s",
                frame_range: "50 - 59",
                frame_dir: "frame_5",
                is_normal: false,
                wrist_distance: 131.45,
                top_classes: [
                  { class: "팔동작(무의미반동)", probability: 82 },
                  { class: "머리동작(고개흔들기)", probability: 12 },
                  { class: "손동작(몸긁기)", probability: 6 }
                ]
              },
              {
                sample_number: 7,
                time_range: "6.00s - 7.00s",
                frame_range: "60 - 69",
                frame_dir: "frame_6",
                is_normal: false,
                wrist_distance: 147.63,
                top_classes: [
                  { class: "자세(비비꼬기)", probability: 75 },
                  { class: "손동작(몸긁기)", probability: 20 },
                  { class: "머리동작(숙이기)", probability: 5 }
                ]
              },
              {
                sample_number: 8,
                time_range: "7.00s - 8.00s",
                frame_range: "70 - 79",
                frame_dir: "frame_7",
                is_normal: true,
                wrist_distance: 122.18,
                top_classes: [
                  { class: "정상", probability: 96 }
                ]
              },
              {
                sample_number: 9,
                time_range: "8.00s - 9.00s",
                frame_range: "80 - 89",
                frame_dir: "frame_8",
                is_normal: false,
                wrist_distance: 135.67,
                top_classes: [
                  { class: "손동작(얼굴)", probability: 88 },
                  { class: "팔동작(뒷짐)", probability: 8 },
                  { class: "머리동작(좌우흔들기)", probability: 4 }
                ]
              },
              {
                sample_number: 10,
                time_range: "9.00s - 10.00s",
                frame_range: "90 - 99",
                frame_dir: "frame_9",
                is_normal: true,
                wrist_distance: 118.93,
                top_classes: [
                  { class: "정상", probability: 91 }
                ]
              },
              {
                sample_number: 11,
                time_range: "10.00s - 11.00s",
                frame_range: "100 - 109",
                frame_dir: "frame_10",
                is_normal: false,
                wrist_distance: 142.52,
                top_classes: [
                  { class: "팔동작(무의미반동)", probability: 79 },
                  { class: "자세(비스듬히)", probability: 15 },
                  { class: "손동작(머리)", probability: 6 }
                ]
              },
              {
                sample_number: 12,
                time_range: "11.00s - 12.00s",
                frame_range: "110 - 119",
                frame_dir: "frame_11",
                is_normal: true,
                wrist_distance: 119.34,
                top_classes: [
                  { class: "정상", probability: 97 }
                ]
              }
            ],
            message: "비언어적 행동 분석이 완료되었습니다."
          };
          
          setAnalysisData(mockData.nonverbal_analysis);
          processAnalysisData(mockData.nonverbal_analysis);
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
    
    fetchNonverbalData();
  }, []);
  
  /**
   * 분석 데이터 처리 함수
   * 
   * 1. 카테고리별 발생 빈도 계산
   * 2. 기본 선택 구간 설정 (비정상 구간 중 첫 번째)
   */
  const processAnalysisData = useCallback((data: NonverbalAnalysisData) => {
    if (!data || data.length === 0) return;
    
    // 카테고리별 발생 빈도 계산
    const categoryCounts: Record<string, number> = {
      '손동작': 0,
      '머리동작': 0,
      '팔동작': 0,
      '자세': 0
    };
    
    // 첫 번째 비정상 구간 인덱스를 찾기 위한 변수
    let firstAbnormalSegmentIndex: number | null = null;
    
    // 각 구간별로 처리
    data.forEach((segment, index) => {
      // 정상 구간이 아닌 경우만 카테고리 집계
      if (!segment.is_normal && segment.top_classes.length > 0) {
        const topClass = segment.top_classes[0].class;
        
        // 첫 번째 비정상 구간 인덱스 저장
        if (firstAbnormalSegmentIndex === null) {
          firstAbnormalSegmentIndex = index;
        }
        
        // 카테고리 접두어 추출 (예: "손동작(머리)" -> "손동작")
        for (const category of Object.keys(categoryCounts)) {
          if (topClass.startsWith(category)) {
            categoryCounts[category] += 1;
            break;
          }
        }
      }
    });
    
    // 차트 데이터 형식으로 변환
    const chartData: CategoryCount[] = Object.entries(categoryCounts).map(([category, count]) => ({
      category,
      count
    }));
    
    setCategoryCountData(chartData);
    
    // 첫 번째 비정상 구간이 있으면 선택
    if (firstAbnormalSegmentIndex !== null) {
      handleSegmentSelect(firstAbnormalSegmentIndex);
    }
  }, []);
  
  /**
   * 구간 선택 핸들러
   * 
   * @param index 선택한 구간의 인덱스
   */
  const handleSegmentSelect = (index: number) => {
    if (!analysisData || !analysisData[index]) return;
    
    const segment = analysisData[index];
    setSelectedSegmentIndex(index);
    
    // 정상 구간인 경우
    if (segment.is_normal) {
      setSelectedBehavior("정상");
      setSelectedFeedback("이 구간에서는 특별한 문제가 감지되지 않았습니다. 좋은 자세와 제스처를 유지하고 있습니다.");
    } 
    // 비정상 구간인 경우
    else if (segment.top_classes.length > 0) {
      const topBehavior = segment.top_classes[0].class;
      setSelectedBehavior(topBehavior);
      
      // 해당 행동에 대한 피드백 가져오기
      const feedback = BEHAVIOR_FEEDBACK[topBehavior] || "이 행동에 대한 구체적인 피드백이 없습니다.";
      setSelectedFeedback(feedback);
    }
    
    // 피드백 영역 갱신 애니메이션
    setShowFeedback(false);
    setTimeout(() => {
      setShowFeedback(true);
    }, 300);
  };
  
  /**
   * 특정 행동 카테고리에 해당하는 색상 반환
   * 
   * @param category 행동 카테고리명
   * @returns 해당 카테고리의 색상 코드
   */
  const getCategoryColor = (category: string): string => {
    return BEHAVIOR_CATEGORIES[category] || '#888888';
  };
  
  /**
   * 대시보드로 돌아가기 함수
   */
  const handleNavigateBack = () => {
    navigate('/analysis');
  };

  /**
   * 구간의 상대적 위치에 따른 썸네일 이미지 URL 생성
   * 
   * 실제 구현 시 백엔드에서 제공하는 썸네일 URL 또는 
   * 비디오 시간에 따른 썸네일 생성 로직으로 대체해야 합니다.
   */
  const getSegmentThumbnail = (segment: NonverbalSegment): string => {
    // 실제 구현 시 frame_dir을 사용하여 이미지 경로 생성
    // return `/api/analysis/frames/${segment.frame_dir}/thumbnail.jpg`;
    
    // 개발 테스트용 더미 이미지 URL
    return `/api/placeholder/640/360?text=구간 ${segment.sample_number} (${segment.time_range})`;
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
              비언어적 분석
            </Typography>
            <Typography 
              variant="body1" 
              color="text.secondary"
              sx={{ mt: 1 }}
            >
              발표 중 자세, 손동작, 머리 움직임 등의 비언어적 요소 분석 결과입니다.
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
            {/* 왼쪽 컬럼: 비디오 썸네일 + 타임라인 */}
            <Grid item xs={12} md={6}>
              <Slide direction="right" in={showContent} timeout={800}>
                <Paper 
                  elevation={0}
                  sx={{ 
                    p: 3, 
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
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="h6" component="h2" fontWeight={600}>
                      발표 영상 구간별 분석
                    </Typography>
                    <VideoLibraryIcon />
                  </Box>
                  
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                    타임라인에서 특정 구간을 선택하여 상세한 비언어적 분석 결과를 확인하세요.
                  </Typography>
                  
                  {loading ? (
                    <Box sx={{ flexGrow: 1, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                      <CircularProgress size={40} />
                    </Box>
                  ) : (
                    <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
                      {/* 비디오 썸네일 영역 */}
                      <Box 
                        sx={{ 
                          width: '100%', 
                          pb: '56.25%', // 16:9 비율 유지
                          position: 'relative',
                          bgcolor: '#f5f5f5',
                          borderRadius: 2,
                          mb: 3,
                          overflow: 'hidden',
                          flexGrow: 0
                        }}
                      >
                        {selectedSegmentIndex !== null && analysisData && (
                          <Box
                            component="img"
                            src={getSegmentThumbnail(analysisData[selectedSegmentIndex])}
                            alt={`발표 구간 ${analysisData[selectedSegmentIndex].sample_number} 썸네일`}
                            sx={{
                              position: 'absolute',
                              top: 0,
                              left: 0,
                              width: '100%',
                              height: '100%',
                              objectFit: 'cover'
                            }}
                          />
                        )}
                        
                        {/* 선택된 구간 정보 오버레이 */}
                        {selectedSegmentIndex !== null && analysisData && (
                          <Box
                            sx={{
                              position: 'absolute',
                              bottom: 0,
                              left: 0,
                              width: '100%',
                              bgcolor: 'rgba(0, 0, 0, 0.7)',
                              color: 'white',
                              p: 2
                            }}
                          >
                            <Typography variant="body2" fontWeight={500}>
                              구간: {analysisData[selectedSegmentIndex].time_range}
                            </Typography>
                            <Typography variant="caption">
                              {analysisData[selectedSegmentIndex].is_normal 
                                ? '정상적인 자세와 제스처' 
                                : `감지된 행동: ${selectedBehavior}`}
                            </Typography>
                          </Box>
                        )}
                      </Box>
                      
                      {/* 타임라인 영역 */}
                      <Typography variant="subtitle2" fontWeight={600} sx={{ mb: 1 }}>
                        타임라인
                      </Typography>

                      {/* 타임라인 스크롤 컨트롤 */}
                      <Box sx={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        mb: 1,
                        justifyContent: 'space-between'
                      }}>
                        <IconButton 
                          size="small" 
                          onClick={() => scrollTimeline('left')}
                          aria-label="타임라인 왼쪽으로 이동"
                        >
                          <ChevronLeftIcon />
                        </IconButton>
                        
                        <Typography variant="caption" color="text.secondary">
                          좌우로 스크롤하여 더 많은 구간 확인
                        </Typography>
                        
                        <IconButton 
                          size="small" 
                          onClick={() => scrollTimeline('right')}
                          aria-label="타임라인 오른쪽으로 이동"
                        >
                          <ChevronRightIcon />
                        </IconButton>
                      </Box>
                      
                      <Box
                        sx={{ 
                          position: 'relative',
                          width: '100%',
                          overflow: 'hidden'
                        }}
                      >
                        <Box 
                          ref={timelineRef}
                          sx={{ 
                            display: 'flex',
                            overflowX: 'auto',
                            width: '100%',
                            scrollbarWidth: 'thin',
                            '&::-webkit-scrollbar': {
                              height: '6px',
                            },
                            '&::-webkit-scrollbar-track': {
                              backgroundColor: '#f1f1f1',
                              borderRadius: '10px',
                            },
                            '&::-webkit-scrollbar-thumb': {
                              backgroundColor: '#888',
                              borderRadius: '10px',
                            },
                            pb: 1
                          }}
                        >
                          {analysisData && analysisData.map((segment, index) => (
                            <Box
                              key={index}
                              onClick={() => handleSegmentSelect(index)}
                              sx={{
                                width: '100px', // 구간별 고정 너비
                                minWidth: '100px',
                                height: '60px',
                                bgcolor: segment.is_normal ? '#4caf50' : '#ff8c00',
                                opacity: selectedSegmentIndex === index ? 1 : 0.7,
                                cursor: 'pointer',
                                position: 'relative',
                                transition: 'all 0.2s ease',
                                mr: 0.5,
                                '&:hover': {
                                  opacity: 0.9,
                                  transform: 'translateY(-2px)'
                                },
                                border: selectedSegmentIndex === index 
                                  ? '2px solid #000' 
                                  : '1px solid rgba(255,255,255,0.3)',
                                borderRadius: 1,
                                display: 'flex',
                                flexDirection: 'column',
                                justifyContent: 'flex-end',
                                alignItems: 'center'
                              }}
                              role="button"
                              aria-label={`발표 구간 ${segment.time_range}`}
                              tabIndex={0}
                            >
                              <Typography 
                                variant="caption" 
                                sx={{ 
                                  color: 'white', 
                                  textAlign: 'center',
                                  fontSize: '10px',
                                  mb: 0.5,
                                  fontWeight: selectedSegmentIndex === index ? 'bold' : 'normal'
                                }}
                              >
                                {index + 1}
                              </Typography>
                              <Typography 
                                variant="caption" 
                                sx={{ 
                                  color: 'white', 
                                  textAlign: 'center',
                                  fontSize: '8px',
                                  bgcolor: 'rgba(0,0,0,0.3)',
                                  px: 0.5,
                                  py: 0.25,
                                  borderRadius: 0.5,
                                  width: '90%'
                                }}
                              >
                                {segment.time_range}
                              </Typography>
                            </Box>
                          ))}
                        </Box>
                      </Box>
                      
                      {/* 타임라인 범례 */}
                      <Box
                        sx={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          mt: 2,
                          gap: 3
                        }}
                      >
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <Box
                            sx={{
                              width: 16,
                              height: 16,
                              borderRadius: 1,
                              bgcolor: '#4caf50',
                              mr: 1
                            }}
                          />
                          <Typography variant="caption">정상 구간</Typography>
                        </Box>
                        <Box sx={{ display: 'flex', alignItems: 'center' }}>
                          <Box
                            sx={{
                              width: 16,
                              height: 16,
                              borderRadius: 1,
                              bgcolor: '#ff8c00',
                              mr: 1
                            }}
                          />
                          <Typography variant="caption">개선 필요 구간</Typography>
                        </Box>
                      </Box>
                    </Box>
                  )}
                </Paper>
              </Slide>
            </Grid>
            
            {/* 오른쪽 컬럼: 분석 정보 */}
            <Grid item xs={12} md={6}>
              <Slide direction="left" in={showContent} timeout={800}>
                <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', gap: 3 }}>
                  {/* 상단: 행동 카테고리 차트 */}
                  <Paper 
                    elevation={0}
                    sx={{ 
                      p: 3, 
                      borderRadius: 3,
                      boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
                      position: 'relative',
                      overflow: 'hidden',
                      flex: '0 0 auto',
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
                        비언어적 행동 발생 빈도
                      </Typography>
                      <CategoryIcon />
                    </Box>
                    
                    <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                      발표 중 감지된 개선이 필요한 비언어적 행동 카테고리별 발생 빈도입니다.
                    </Typography>
                    
                    {loading ? (
                      <Box sx={{ height: 240, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                        <CircularProgress size={40} />
                      </Box>
                    ) : (
                      <Box sx={{ height: 240, width: '100%' }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={categoryCountData}
                            margin={{ top: 10, right: 10, left: 10, bottom: 20 }}
                          >
                            <CartesianGrid strokeDasharray="3 3" vertical={false} />
                            <XAxis dataKey="category" />
                            <YAxis allowDecimals={false} />
                            <Tooltip
                              formatter={(value) => [`${value}회`, '발생 빈도']}
                              contentStyle={{ 
                                backgroundColor: '#fff', 
                                borderRadius: '8px',
                                boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                                border: 'none' 
                              }}
                            />
                            <Bar dataKey="count" name="발생 빈도">
                              {categoryCountData.map((entry, index) => (
                                <Cell 
                                  key={`cell-${index}`} 
                                  fill={getCategoryColor(entry.category)} 
                                />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </Box>
                    )}
                  </Paper>
                  
                  {/* 중간: 선택된 구간 피드백 */}
                  <Paper 
                    elevation={0}
                    sx={{ 
                      p: 3, 
                      borderRadius: 3,
                      boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
                      position: 'relative',
                      overflow: 'hidden',
                      flex: '1 1 auto',
                      display: 'flex',
                      flexDirection: 'column',
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
                    <Typography variant="h6" component="h2" fontWeight={600} sx={{ mb: 3 }}>
                      선택 구간 피드백
                    </Typography>
                    
                    {selectedSegmentIndex !== null && analysisData ? (
                      <Box sx={{ flexGrow: 1 }}>
                        <Box sx={{ mb: 3 }}>
                          <Typography variant="subtitle1" fontWeight={600} gutterBottom>
                            {`선택 구간: ${analysisData[selectedSegmentIndex].time_range}`}
                          </Typography>
                          <Box 
                            sx={{ 
                              display: 'flex', 
                              alignItems: 'center',
                              p: 2,
                              bgcolor: analysisData[selectedSegmentIndex].is_normal ? '#e8f5e9' : '#fff3e0',
                              borderRadius: 2
                            }}
                          >
                            <Typography 
                              variant="body2" 
                              fontWeight={500}
                              color={analysisData[selectedSegmentIndex].is_normal ? 'success.dark' : 'warning.dark'}
                            >
                              상태: {analysisData[selectedSegmentIndex].is_normal ? '정상' : '개선 필요'}
                            </Typography>
                          </Box>
                        </Box>
                        
                        <Divider sx={{ my: 2 }} />
                        
                        <Fade in={showFeedback} timeout={500}>
                          <Box>
                            {!analysisData[selectedSegmentIndex].is_normal && (
                              <Box sx={{ mb: 3 }}>
                                <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                                  감지된 행동
                                </Typography>
                                <Box 
                                  sx={{ 
                                    p: 2, 
                                    borderRadius: 2, 
                                    bgcolor: '#f5f5f5',
                                    border: '1px solid #eee'
                                  }}
                                >
                                  <Typography variant="body1" fontWeight={500}>
                                    {selectedBehavior}
                                  </Typography>
                                  <Typography variant="caption" color="text.secondary">
                                    확률: {analysisData[selectedSegmentIndex].top_classes[0].probability}%
                                  </Typography>
                                </Box>
                              </Box>
                            )}
                            
                            <Typography variant="subtitle2" fontWeight={600} gutterBottom>
                              피드백
                            </Typography>
                            <Paper 
                              elevation={0}
                              sx={{ 
                                p: 3, 
                                borderRadius: 3,
                                bgcolor: '#fff',
                                border: '1px solid #eaeaea',
                                boxShadow: '0 2px 10px rgba(0,0,0,0.05)'
                              }}
                            >
                              <Typography variant="body1">
                                {selectedFeedback}
                              </Typography>
                            </Paper>
                          </Box>
                        </Fade>
                      </Box>
                    ) : (
                      <Box 
                        sx={{ 
                          display: 'flex', 
                          flexDirection: 'column',
                          justifyContent: 'center', 
                          alignItems: 'center',
                          flexGrow: 1,
                          p: 3,
                          textAlign: 'center'
                        }}
                      >
                        <InfoIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
                        <Typography>
                          타임라인에서 구간을 선택하면 해당 구간의 분석 결과가 여기에 표시됩니다.
                        </Typography>
                      </Box>
                    )}
                  </Paper>
                </Box>
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
            onClick={() => navigate('/analysis/script')}
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
            대본 분석 보기
          </Button>
        </Box>
        
        {/* 푸터 */}
        <Box 
          component="footer"
          sx={{ 
            mt: 4, 
            textAlign: 'center', 
            borderTop: '1px solid #eee',
            pt: 3,
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

export default NonverbalEvaluationDetailPage;
