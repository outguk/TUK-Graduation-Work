// /**
//  * NonverbalEvaluationDetailPage.tsx
//  * 
//  * 설명: 발표 분석의 비언어적 요소 세부 결과 페이지
//  * 
//  * URL 파라미터:
//  * - presentationId: 발표 ID (optional, 없으면 가장 최근 발표 분석 표시)
//  * 
//  * 백엔드 API 연동 포인트:
//  * - GET /api/presentations/{presentationId}/nonverbal
//  *   : 특정 발표의 비언어적 분석 데이터 조회
//  * 
//  * 기능:
//  * 1. 타임라인 기반 비디오 세그먼트 표시 및 상호작용
//  * 2. 비언어적 행동 카테고리별 발생 빈도 차트
//  * 3. 선택된 구간의 비언어적 행동 피드백 표시
//  * 
//  * 응답 형식은 mongoDB의 nonverbal_analysis형식과 동일
//  */

// import React, { useState, useEffect, useCallback, useRef } from 'react';
// import {
//   Box,
//   Container,
//   Typography,
//   Grid,
//   Paper,
//   Button,
//   Fade,
//   Slide,
//   IconButton,
//   Alert,
//   CircularProgress,
//   Divider,
// } from '@mui/material';
// import {
//   BarChart,
//   Bar,
//   XAxis,
//   YAxis,
//   CartesianGrid,
//   Tooltip,
//   ResponsiveContainer,
//   Cell,
// } from 'recharts';
// import { useNavigate, useParams } from 'react-router-dom';
// import {
//   ArrowBack as ArrowBackIcon,
//   Info as InfoIcon,
//   VideoLibrary as VideoLibraryIcon,
//   Category as CategoryIcon,
//   ChevronLeft as ChevronLeftIcon,
//   ChevronRight as ChevronRightIcon,
// } from '@mui/icons-material';


// /**
//  * 백엔드 데이터 모델과 일치하는 타입 정의
//  * 
//  * NonverbalSegment: 
//  * - 비언어적 분석 결과의 각 시간 구간 데이터
//  * - sample_number: 샘플 번호 (1부터 시작)
//  * - time_range: 시간 범위 (예: "0.00s - 1.00s")
//  * - is_normal: 정상 행동 여부
//  * - top_classes: 감지된 행동 클래스와 확률
//  * 
//  * 백엔드 개발자 참고:
//  * - is_normal이 false인 경우 top_classes 배열에 감지된 행동 정보가 필요함
//  * - top_classes[0]는 가장 높은 확률의 행동이어야 함
//  */

// // Import mock data
// import { 
//   mockAnalysisDataMap, 
//   PresentationAnalysis,
//   NonverbalSegment
// } from '../components/mockAnalysisData';

// /**
//  * 데이터 타입 정의
//  */
// // 행동 카테고리 발생 빈도 타입
// interface CategoryCount {
//   category: string;
//   count: number;
// }

// // 행동 카테고리 정의 및 색상 매핑
// const BEHAVIOR_CATEGORIES: { [key: string]: string } = {
//   '자세': '#63e6be',
//   '손동작': '#ff6b6b',
//   '머리동작': '#4dabf7',
//   '팔동작': '#4dabf7'
// };

// // 행동별 피드백 매핑 (행동 -> 개선 팁)
// const BEHAVIOR_FEEDBACK: { [key: string]: string } = {
//   "손동작(머리)": "머리를 만지는 습관은 불안한 인상을 줄 수 있어요. 손은 안정된 위치에 두는 것이 좋아요.",
//   "손동작(얼굴)": "얼굴을 만지는 행동은 산만해 보일 수 있어요. 청중과의 시선 유지에 집중해보세요.",
//   "손동작(몸긁기)": "몸을 긁는 동작은 불편함을 드러낼 수 있어요. 손은 가볍게 모으거나 제스처로 활용해보세요.",
//   "손동작(손톱)": "손톱을 만지거나 뜯는 동작은 긴장감을 전달해요. 손을 자연스럽게 두는 연습이 필요해요.",
//   "머리동작(고개흔들기)": "고개를 자주 흔드는 것은 산만한 인상을 줄 수 있어요. 시선과 자세를 고정해보세요.",
//   "머리동작(좌우흔들기)": "머리를 좌우로 흔드는 습관은 주의를 흩뜨릴 수 있어요. 차분한 고개 움직임을 유지하세요.",
//   "머리동작(숙이기)": "고개를 숙이는 자세는 자신감 부족으로 보일 수 있어요. 시선을 정면으로 유지해보세요.",
//   "팔동작(뒷짐)": "뒷짐은 청중과의 거리감을 줄 수 있어요. 앞에 두고 자연스러운 제스처를 사용하는 것이 좋습니다.",
//   "팔동작(무의미반동)": "불필요한 팔의 움직임은 발표 흐름을 방해할 수 있어요. 의미 있는 제스처만 사용하는 연습이 필요해요.",
//   "자세(좌우흔들기)": "몸을 좌우로 흔드는 습관은 발표자의 긴장을 드러냅니다. 중심을 잡고 안정된 자세를 유지해보세요.",
//   "자세(비스듬히)": "비스듬한 자세는 집중이 부족해 보일 수 있어요. 정면을 향한 단단한 자세가 신뢰감을 줍니다.",
//   "자세(비비꼬기)": "다리를 꼬는 자세는 불안정하고 산만한 인상을 줄 수 있어요. 두 다리를 안정적으로 두는 자세를 연습하세요."
// };

// const NonverbalEvaluationDetailPage: React.FC = () => {
//   const navigate = useNavigate();
//   const timelineRef = useRef<HTMLDivElement>(null);
  
//   // URL 파라미터에서 presentationId 추출
//   const { presentationId } = useParams<{ presentationId?: string }>();
  
//   // 상태 관리
//   const [presentation, setPresentation] = useState<PresentationAnalysis | null>(null);
//   const [analysisData, setAnalysisData] = useState<NonverbalSegment[] | null>(null);
//   const [categoryCountData, setCategoryCountData] = useState<CategoryCount[]>([]);
//   const [selectedSegmentIndex, setSelectedSegmentIndex] = useState<number | null>(null);
//   const [selectedBehavior, setSelectedBehavior] = useState<string>("");
//   const [selectedFeedback, setSelectedFeedback] = useState<string>("구간을 선택하면 상세 피드백이 표시됩니다.");
//   const [loading, setLoading] = useState(true);
//   const [error, setError] = useState<string | null>(null);
  
//   // UI 애니메이션 상태
//   const [showContent, setShowContent] = useState(false);
//   const [showFeedback, setShowFeedback] = useState(false);
  
//   // 페이지 로딩 시 애니메이션 설정
//   useEffect(() => {
//     const timer = setTimeout(() => setShowContent(true), 300);
//     return () => clearTimeout(timer);
//   }, []);
  
//   /**
//    * 타임라인 좌우 스크롤 함수
//    */
//   const scrollTimeline = (direction: 'left' | 'right') => {
//     if (timelineRef.current) {
//       const scrollAmount = 200; // 스크롤 양 (픽셀)
//       const currentScroll = timelineRef.current.scrollLeft;
      
//       timelineRef.current.scrollTo({
//         left: direction === 'left' ? currentScroll - scrollAmount : currentScroll + scrollAmount,
//         behavior: 'smooth'
//       });
//     }
//   };

//   /**
//  * 백엔드 API 호출 구현 필요
//  * 
//  * 현재는 mockAnalysisDataMap에서 데이터를 가져오지만,
//  * 실제 구현 시 아래 API를 호출해야 함:
//  * 
//  * GET /api/presentations/{presentationId}/nonverbal
//  * 
//  * 백엔드 개발자 참고:
//  * 1. presentationId가 URL 파라미터로 전달됨
//  * 2. 응답은 NonverbalAnalysisResult 타입과 일치해야 함
//  * 3. 에러 처리를 위한 적절한 HTTP 상태 코드 반환 필요
//  *    - 404: 발표를 찾을 수 없음
//  *    - 403: 접근 권한 없음
//  *    - 500: 서버 오류
//  * 
//  * 현재는 mockAnalysisDataMap에서 발표 ID에 해당하는 비언어적 분석 데이터 가져옴
//  */
//   useEffect(() => {
//     const fetchNonverbalData = async () => {
//       setLoading(true);
//       setError(null);
      
//       try {
//       // 실제 API 호출 코드로 대체 필요
//       // const response = await axios.get<NonverbalAnalysisResult>(
//       //   `/api/presentations/${presentationId}/nonverbal`
//       // );
//       // setPresentation(response.data.presentation);
//       // setAnalysisData(response.data.nonverbal_analysis);
      
//     // 목업 데이터 (백엔드 개발 완료 후 제거)  
//         // 발표 ID에 해당하는 데이터 가져오기
//         let selectedPresentation: PresentationAnalysis | null = null;
        
//         if (presentationId && mockAnalysisDataMap[presentationId]) {
//           // ID에 해당하는 발표 데이터 가져오기
//           selectedPresentation = mockAnalysisDataMap[presentationId];
//         } else {
//           // ID가 없거나 해당하는 발표가 없으면 첫 번째 발표 데이터 사용
//           const firstPresentationId = Object.keys(mockAnalysisDataMap)[0];
//           selectedPresentation = mockAnalysisDataMap[firstPresentationId];
//         }
        
//         if (selectedPresentation) {
//           setPresentation(selectedPresentation);
//           setAnalysisData(selectedPresentation.nonverbal_analysis);
//           processAnalysisData(selectedPresentation.nonverbal_analysis);
//         } else {
//           throw new Error("발표 데이터를 찾을 수 없습니다.");
//         }
//       } catch (err) {
//         console.error("발표 데이터 로드 에러:", err);
//         setError('데이터를 불러오는 중 오류가 발생했습니다.');
//       } finally {
//         setLoading(false);
//       }
//     };
    
//     fetchNonverbalData();
//   }, [presentationId]);
  
//  /**
//  * 분석 데이터 처리 함수
//  * 
//  * 백엔드 개발자 참고:
//  * 1. 비언어적 행동 카테고리는 접두어로 구분함:
//  *    - '손동작': 손동작(머리), 손동작(얼굴) 등
//  *    - '머리동작': 머리동작(고개흔들기), 머리동작(숙이기) 등
//  *    - '팔동작': 팔동작(뒷짐), 팔동작(무의미반동) 등
//  *    - '자세': 자세(비비꼬기), 자세(좌우흔들기) 등
//  * 
//  * 2. 백엔드에서 행동 클래스명은 위 패턴을 따라야 함
//  * 3. is_normal이 true이면 top_classes는 빈 배열이거나 "정상"만 포함
//  */
//   const processAnalysisData = useCallback((data: NonverbalSegment[]) => {
//     if (!data || data.length === 0) return;
    
//     // 카테고리별 발생 빈도 계산
//     const categoryCounts: Record<string, number> = {
//       '손동작': 0,
//       '머리동작': 0,
//       '팔동작': 0,
//       '자세': 0
//     };
    
//     // 첫 번째 비정상 구간 인덱스를 찾기 위한 변수
//     let firstAbnormalSegmentIndex: number | null = null;
    
//     // 각 구간별로 처리
//     data.forEach((segment, index) => {
//       // 정상 구간이 아닌 경우만 카테고리 집계
//       if (!segment.is_normal && segment.top_classes && segment.top_classes.length > 0) {
//         const topClass = segment.top_classes[0].class;
        
//         // 첫 번째 비정상 구간 인덱스 저장
//         if (firstAbnormalSegmentIndex === null) {
//           firstAbnormalSegmentIndex = index;
//         }
        
//         // 카테고리 접두어 추출 (예: "손동작(머리)" -> "손동작")
//         for (const category of Object.keys(categoryCounts)) {
//           if (topClass.startsWith(category)) {
//             categoryCounts[category] += 1;
//             break;
//           }
//         }
//       }
//     });
    
//     // 차트 데이터 형식으로 변환
//     const chartData: CategoryCount[] = Object.entries(categoryCounts).map(([category, count]) => ({
//       category,
//       count
//     }));
    
//     setCategoryCountData(chartData);
    
//     // 첫 번째 비정상 구간이 있으면 선택
//     if (firstAbnormalSegmentIndex !== null) {
//       handleSegmentSelect(firstAbnormalSegmentIndex);
//     }
//   }, []);
  
//   /**
//    * 구간 선택 핸들러
//    * 
//    * @param index 선택한 구간의 인덱스
//    */
//   const handleSegmentSelect = (index: number) => {
//     if (!analysisData || !analysisData[index]) return;
    
//     const segment = analysisData[index];
//     setSelectedSegmentIndex(index);
    
//     // 정상 구간인 경우
//     if (segment.is_normal) {
//       setSelectedBehavior("정상");
//       setSelectedFeedback("이 구간에서는 특별한 문제가 감지되지 않았습니다. 좋은 자세와 제스처를 유지하고 있습니다.");
//     } 
//     // 비정상 구간인 경우
//     else if (segment.top_classes && segment.top_classes.length > 0) {
//       const topBehavior = segment.top_classes[0].class;
//       setSelectedBehavior(topBehavior);
      
//       // 해당 행동에 대한 피드백 가져오기
//       const feedback = BEHAVIOR_FEEDBACK[topBehavior] || "이 행동에 대한 구체적인 피드백이 없습니다.";
//       setSelectedFeedback(feedback);
//     }
    
//     // 피드백 영역 갱신 애니메이션
//     setShowFeedback(false);
//     setTimeout(() => {
//       setShowFeedback(true);
//     }, 300);
//   };
  
//   /**
//    * 특정 행동 카테고리에 해당하는 색상 반환
//    * 
//    * @param category 행동 카테고리명
//    * @returns 해당 카테고리의 색상 코드
//    */
//   const getCategoryColor = (category: string): string => {
//     return BEHAVIOR_CATEGORIES[category] || '#888888';
//   };
  
//   /**
//    * 대시보드로 돌아가기 함수
//    * 만약 presentationId가 있으면 해당 ID를 유지한 채 대시보드로 이동
//    */
//   const handleNavigateBack = () => {
//     if (presentationId) {
//       navigate(`/analysis/${presentationId}`);
//     } else {
//       navigate('/analysis');
//     }
//   };

//   /**
//    * 구간의 상대적 위치에 따른 썸네일 이미지 URL 생성
//    * 
//    * 실제 구현 시 백엔드에서 제공하는 썸네일 URL 또는 
//    * 비디오 시간에 따른 썸네일 생성 로직으로 대체해야 합니다.
//    * frame_dir 필드를 사용하여 이미지 URL 생성 예정
//    */
  
//   const getSegmentThumbnail = (segment: NonverbalSegment): string => {
//     // 실제 구현 시 frame_dir을 사용하여 이미지 경로 생성
//     // return `/api/analysis/frames/${segment.frame_dir}/thumbnail.jpg`;
    
//     // 개발 테스트용 더미 이미지 URL
//     return `/api/placeholder/640/360?text=구간 ${segment.sample_number} (${segment.time_range})`;
//   };

//   return (
//     <Box
//       component="main"
//       sx={{
//         minHeight: '100vh',
//         py: { xs: 4, md: 6 },
//         px: { xs: 2, md: 4 },
//         background: '#FFFFFF',
//       }}
//     >
//       <Container maxWidth="xl">
//         {/* 헤더 섹션 */}
//         <Box sx={{ mb: 5, display: 'flex', alignItems: 'center' }}>
//           <IconButton 
//             aria-label="돌아가기" 
//             onClick={handleNavigateBack}
//             sx={{ mr: 2 }}
//           >
//             <ArrowBackIcon />
//           </IconButton>
//           <Box>
//             <Typography 
//               variant="h4" 
//               component="h1"
//               fontWeight={700}
//               sx={{ 
//                 mb: 1,
//                 position: 'relative',
//                 display: 'inline-block'
//               }}
//             >
//               비언어적 분석
//               {presentation && ` - ${presentation.title}`}
//             </Typography>
//             <Typography 
//               variant="body1" 
//               color="text.secondary"
//               sx={{ mt: 1 }}
//             >
//               발표 중 자세, 손동작, 머리 움직임 등의 비언어적 요소 분석 결과입니다.
//             </Typography>
//           </Box>
//         </Box>
        
//         {/* 에러 메시지 표시 */}
//         {error && (
//           <Alert 
//             severity="error" 
//             sx={{ mb: 3 }}
//             onClose={() => setError(null)}
//           >
//             {error}
//           </Alert>
//         )}
        
//         <Fade in={showContent} timeout={1000}>
//           <Grid container spacing={4}>

//             {/* 
//             비디오 썸네일 영역
            
//             백엔드 개발자 참고:
//             1. 각 구간별 대표 이미지가 필요함 (시각적 피드백을 위해 중요)
//             2. segment.frame_dir 필드에 이미지 디렉토리 경로가 들어있어야 함
//             3. 썸네일 이미지 크기는 640x360px 권장
//             */}

//             {/* 왼쪽 컬럼: 비디오 썸네일 + 타임라인 */}
//             <Grid item xs={12} md={6}>
//               <Slide direction="right" in={showContent} timeout={800}>
//                 <Paper 
//                   elevation={0}
//                   sx={{ 
//                     p: 3, 
//                     borderRadius: 3,
//                     boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
//                     height: '100%',
//                     display: 'flex',
//                     flexDirection: 'column',
//                     position: 'relative',
//                     overflow: 'hidden',
//                     '&::before': {
//                       content: '""',
//                       position: 'absolute',
//                       top: 0,
//                       left: 0,
//                       width: '100%',
//                       height: '4px',
//                       background: '#000'
//                     }
//                   }}
//                 >
//                   <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
//                     <Typography variant="h6" component="h2" fontWeight={600}>
//                       발표 영상 구간별 분석
//                     </Typography>
//                     <VideoLibraryIcon />
//                   </Box>
                  
//                   <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
//                     타임라인에서 특정 구간을 선택하여 상세한 비언어적 분석 결과를 확인하세요.
//                   </Typography>
                  
//                   {loading ? (
//                     <Box sx={{ flexGrow: 1, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
//                       <CircularProgress size={40} />
//                     </Box>
//                   ) : (
//                     <Box sx={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
//                       {/* 비디오 썸네일 영역 */}
//                       <Box 
//                         sx={{ 
//                           width: '100%', 
//                           pb: '56.25%', // 16:9 비율 유지
//                           position: 'relative',
//                           bgcolor: '#f5f5f5',
//                           borderRadius: 2,
//                           mb: 3,
//                           overflow: 'hidden',
//                           flexGrow: 0
//                         }}
//                       >
//                         {selectedSegmentIndex !== null && analysisData && (
//                           <Box
//                             component="img"
//                             src={getSegmentThumbnail(analysisData[selectedSegmentIndex])}
//                             alt={`발표 구간 ${analysisData[selectedSegmentIndex].sample_number} 썸네일`}
//                             sx={{
//                               position: 'absolute',
//                               top: 0,
//                               left: 0,
//                               width: '100%',
//                               height: '100%',
//                               objectFit: 'cover'
//                             }}
//                           />
//                         )}
                        
//                         {/* 선택된 구간 정보 오버레이 */}
//                         {selectedSegmentIndex !== null && analysisData && (
//                           <Box
//                             sx={{
//                               position: 'absolute',
//                               bottom: 0,
//                               left: 0,
//                               width: '100%',
//                               bgcolor: 'rgba(0, 0, 0, 0.7)',
//                               color: 'white',
//                               p: 2
//                             }}
//                           >
//                             <Typography variant="body2" fontWeight={500}>
//                               구간: {analysisData[selectedSegmentIndex].time_range}
//                             </Typography>
//                             <Typography variant="caption">
//                               {analysisData[selectedSegmentIndex].is_normal 
//                                 ? '정상적인 자세와 제스처' 
//                                 : `감지된 행동: ${selectedBehavior}`}
//                             </Typography>
//                           </Box>
//                         )}
//                       </Box>
                      
//                       {/* 타임라인 영역 */}
//                       <Typography variant="subtitle2" fontWeight={600} sx={{ mb: 1 }}>
//                         타임라인
//                       </Typography>

//                       {/* 타임라인 스크롤 컨트롤 */}
//                       <Box sx={{ 
//                         display: 'flex', 
//                         alignItems: 'center', 
//                         mb: 1,
//                         justifyContent: 'space-between'
//                       }}>
//                         <IconButton 
//                           size="small" 
//                           onClick={() => scrollTimeline('left')}
//                           aria-label="타임라인 왼쪽으로 이동"
//                         >
//                           <ChevronLeftIcon />
//                         </IconButton>
                        
//                         <Typography variant="caption" color="text.secondary">
//                           좌우로 스크롤하여 더 많은 구간 확인
//                         </Typography>
                        
//                         <IconButton 
//                           size="small" 
//                           onClick={() => scrollTimeline('right')}
//                           aria-label="타임라인 오른쪽으로 이동"
//                         >
//                           <ChevronRightIcon />
//                         </IconButton>
//                       </Box>
                      
//                       <Box
//                         sx={{ 
//                           position: 'relative',
//                           width: '100%',
//                           overflow: 'hidden'
//                         }}
//                       >
//                         <Box 
//                           ref={timelineRef}
//                           sx={{ 
//                             display: 'flex',
//                             overflowX: 'auto',
//                             width: '100%',
//                             scrollbarWidth: 'thin',
//                             '&::-webkit-scrollbar': {
//                               height: '6px',
//                             },
//                             '&::-webkit-scrollbar-track': {
//                               backgroundColor: '#f1f1f1',
//                               borderRadius: '10px',
//                             },
//                             '&::-webkit-scrollbar-thumb': {
//                               backgroundColor: '#888',
//                               borderRadius: '10px',
//                             },
//                             pb: 1
//                           }}
//                         >
//                           {analysisData && analysisData.map((segment, index) => (
//                             <Box
//                               key={index}
//                               onClick={() => handleSegmentSelect(index)}
//                               sx={{
//                                 width: '100px', // 구간별 고정 너비
//                                 minWidth: '100px',
//                                 height: '60px',
//                                 bgcolor: segment.is_normal ? '#4caf50' : '#ff8c00',
//                                 opacity: selectedSegmentIndex === index ? 1 : 0.7,
//                                 cursor: 'pointer',
//                                 position: 'relative',
//                                 transition: 'all 0.2s ease',
//                                 mr: 0.5,
//                                 '&:hover': {
//                                   opacity: 0.9,
//                                   transform: 'translateY(-2px)'
//                                 },
//                                 border: selectedSegmentIndex === index 
//                                   ? '2px solid #000' 
//                                   : '1px solid rgba(255,255,255,0.3)',
//                                 borderRadius: 1,
//                                 display: 'flex',
//                                 flexDirection: 'column',
//                                 justifyContent: 'flex-end',
//                                 alignItems: 'center'
//                               }}
//                               role="button"
//                               aria-label={`발표 구간 ${segment.time_range}`}
//                               tabIndex={0}
//                             >
//                               <Typography 
//                                 variant="caption" 
//                                 sx={{ 
//                                   color: 'white', 
//                                   textAlign: 'center',
//                                   fontSize: '10px',
//                                   mb: 0.5,
//                                   fontWeight: selectedSegmentIndex === index ? 'bold' : 'normal'
//                                 }}
//                               >
//                                 {index + 1}
//                               </Typography>
//                               <Typography 
//                                 variant="caption" 
//                                 sx={{ 
//                                   color: 'white', 
//                                   textAlign: 'center',
//                                   fontSize: '8px',
//                                   bgcolor: 'rgba(0,0,0,0.3)',
//                                   px: 0.5,
//                                   py: 0.25,
//                                   borderRadius: 0.5,
//                                   width: '90%'
//                                 }}
//                               >
//                                 {segment.time_range}
//                               </Typography>
//                             </Box>
//                           ))}
//                         </Box>
//                       </Box>
                      
//                       {/* 타임라인 범례 */}
//                       <Box
//                         sx={{
//                           display: 'flex',
//                           alignItems: 'center',
//                           justifyContent: 'center',
//                           mt: 2,
//                           gap: 3
//                         }}
//                       >
//                         <Box sx={{ display: 'flex', alignItems: 'center' }}>
//                           <Box
//                             sx={{
//                               width: 16,
//                               height: 16,
//                               borderRadius: 1,
//                               bgcolor: '#4caf50',
//                               mr: 1
//                             }}
//                           />
//                           <Typography variant="caption">정상 구간</Typography>
//                         </Box>
//                         <Box sx={{ display: 'flex', alignItems: 'center' }}>
//                           <Box
//                             sx={{
//                               width: 16,
//                               height: 16,
//                               borderRadius: 1,
//                               bgcolor: '#ff8c00',
//                               mr: 1
//                             }}
//                           />
//                           <Typography variant="caption">개선 필요 구간</Typography>
//                         </Box>
//                       </Box>
//                     </Box>
//                   )}
//                 </Paper>
//               </Slide>
//             </Grid>
            
//             {/* 오른쪽 컬럼: 분석 정보 */}
//             <Grid item xs={12} md={6}>
//               <Slide direction="left" in={showContent} timeout={800}>
//                 <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', gap: 3 }}>
//                 {/* 
//                 차트 영역
                
//                 백엔드 개발자 참고:
//                 1. 카테고리별 발생 빈도는 클라이언트에서 계산함 
//                 2. 각 행동의 카테고리는 행동명의 접두어로 판단함
//                 3. top_classes[0].class 값이 "자세(비스듬히)"와 같은 형식이어야 함
//                 */}

//                   {/* 상단: 행동 카테고리 차트 */}
//                   <Paper 
//                     elevation={0}
//                     sx={{ 
//                       p: 3, 
//                       borderRadius: 3,
//                       boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
//                       position: 'relative',
//                       overflow: 'hidden',
//                       flex: '0 0 auto',
//                       '&::before': {
//                         content: '""',
//                         position: 'absolute',
//                         top: 0,
//                         left: 0,
//                         width: '100%',
//                         height: '4px',
//                         background: '#000'
//                       }
//                     }}
//                   >
//                     <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
//                       <Typography variant="h6" component="h2" fontWeight={600}>
//                         비언어적 행동 발생 빈도
//                       </Typography>
//                       <CategoryIcon />
//                     </Box>
                    
//                     <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
//                       발표 중 감지된 개선이 필요한 비언어적 행동 카테고리별 발생 빈도입니다.
//                     </Typography>
                    
//                     {loading ? (
//                       <Box sx={{ height: 240, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
//                         <CircularProgress size={40} />
//                       </Box>
//                     ) : (
//                       <Box sx={{ height: 240, width: '100%' }}>
//                         <ResponsiveContainer width="100%" height="100%">
//                           <BarChart
//                             data={categoryCountData}
//                             margin={{ top: 10, right: 10, left: 10, bottom: 20 }}
//                           >
//                             <CartesianGrid strokeDasharray="3 3" vertical={false} />
//                             <XAxis dataKey="category" />
//                             <YAxis allowDecimals={false} />
//                             <Tooltip
//                               formatter={(value) => [`${value}회`, '발생 빈도']}
//                               contentStyle={{ 
//                                 backgroundColor: '#fff', 
//                                 borderRadius: '8px',
//                                 boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
//                                 border: 'none' 
//                               }}
//                             />
//                             <Bar dataKey="count" name="발생 빈도">
//                               {categoryCountData.map((entry, index) => (
//                                 <Cell 
//                                   key={`cell-${index}`} 
//                                   fill={getCategoryColor(entry.category)} 
//                                 />
//                               ))}
//                             </Bar>
//                           </BarChart>
//                         </ResponsiveContainer>
//                       </Box>
//                     )}
//                   </Paper>
                  
//                 {/* 
//                 피드백 영역
                
//                 백엔드 개발자 참고:
//                 1. 이 영역은 선택된 구간의 상세 피드백을 표시함
//                 2. top_classes[0].probability는 백분율로 표시되어야 함 (0-100)
//                 3. 피드백 텍스트는 클라이언트에서 정의된 매핑 사용 (BEHAVIOR_FEEDBACK)
//                 4. 향후 확장: 백엔드에서 각 행동별 맞춤 피드백 제공 API 개발 가능
//                 */}

//                   {/* 중간: 선택된 구간 피드백 */}
//                   <Paper 
//                     elevation={0}
//                     sx={{ 
//                       p: 3, 
//                       borderRadius: 3,
//                       boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
//                       position: 'relative',
//                       overflow: 'hidden',
//                       flex: '1 1 auto',
//                       display: 'flex',
//                       flexDirection: 'column',
//                       '&::before': {
//                         content: '""',
//                         position: 'absolute',
//                         top: 0,
//                         left: 0,
//                         width: '100%',
//                         height: '4px',
//                         background: '#000'
//                       }
//                     }}
//                   >
//                     <Typography variant="h6" component="h2" fontWeight={600} sx={{ mb: 3 }}>
//                       선택 구간 피드백
//                     </Typography>
                    
//                     {selectedSegmentIndex !== null && analysisData ? (
//                       <Box sx={{ flexGrow: 1 }}>
//                         <Box sx={{ mb: 3 }}>
//                           <Typography variant="subtitle1" fontWeight={600} gutterBottom>
//                             {`선택 구간: ${analysisData[selectedSegmentIndex].time_range}`}
//                           </Typography>
//                           <Box 
//                             sx={{ 
//                               display: 'flex', 
//                               alignItems: 'center',
//                               p: 2,
//                               bgcolor: analysisData[selectedSegmentIndex].is_normal ? '#e8f5e9' : '#fff3e0',
//                               borderRadius: 2
//                             }}
//                           >
//                             <Typography 
//                               variant="body2" 
//                               fontWeight={500}
//                               color={analysisData[selectedSegmentIndex].is_normal ? 'success.dark' : 'warning.dark'}
//                             >
//                               상태: {analysisData[selectedSegmentIndex].is_normal ? '정상' : '개선 필요'}
//                             </Typography>
//                           </Box>
//                         </Box>
                        
//                         <Divider sx={{ my: 2 }} />
                        
//                         <Fade in={showFeedback} timeout={500}>
//                           <Box>
//                             {!analysisData[selectedSegmentIndex].is_normal && analysisData[selectedSegmentIndex].top_classes && (
//                               <Box sx={{ mb: 3 }}>
//                                 <Typography variant="subtitle2" fontWeight={600} gutterBottom>
//                                   감지된 행동
//                                 </Typography>
//                                 <Box 
//                                   sx={{ 
//                                     p: 2, 
//                                     borderRadius: 2, 
//                                     bgcolor: '#f5f5f5',
//                                     border: '1px solid #eee'
//                                   }}
//                                 >
//                                   <Typography variant="body1" fontWeight={500}>
//                                     {selectedBehavior}
//                                   </Typography>
//                                   <Typography variant="caption" color="text.secondary">
//                                     확률: {analysisData[selectedSegmentIndex].top_classes[0].probability}%
//                                   </Typography>
//                                 </Box>
//                               </Box>
//                             )}
                            
//                             <Typography variant="subtitle2" fontWeight={600} gutterBottom>
//                               피드백
//                             </Typography>
//                             <Paper 
//                               elevation={0}
//                               sx={{ 
//                                 p: 3, 
//                                 borderRadius: 3,
//                                 bgcolor: '#fff',
//                                 border: '1px solid #eaeaea',
//                                 boxShadow: '0 2px 10px rgba(0,0,0,0.05)'
//                               }}
//                             >
//                               <Typography variant="body1">
//                                 {selectedFeedback}
//                               </Typography>
//                             </Paper>
//                           </Box>
//                         </Fade>
//                       </Box>
//                     ) : (
//                       <Box 
//                         sx={{ 
//                           display: 'flex', 
//                           flexDirection: 'column',
//                           justifyContent: 'center', 
//                           alignItems: 'center',
//                           flexGrow: 1,
//                           p: 3,
//                           textAlign: 'center'
//                         }}
//                       >
//                         <InfoIcon sx={{ fontSize: 48, color: 'text.secondary', mb: 2 }} />
//                         <Typography>
//                           타임라인에서 구간을 선택하면 해당 구간의 분석 결과가 여기에 표시됩니다.
//                         </Typography>
//                       </Box>
//                     )}
//                   </Paper>
//                 </Box>
//               </Slide>
//             </Grid>
//           </Grid>
//         </Fade>
        
//         {/* 하단 탐색 버튼 영역 */}
//         <Box 
//           sx={{ 
//             display: 'flex', 
//             justifyContent: 'space-between',
//             mt: 4 
//           }}
//         >
//           <Button
//             variant="outlined"
//             startIcon={<ArrowBackIcon />}
//             onClick={handleNavigateBack}
//             sx={{ 
//               borderRadius: 8,
//               px: 3,
//               py: 1,
//               borderColor: '#000',
//               color: '#000',
//               '&:hover': {
//                 borderColor: '#333',
//                 backgroundColor: 'rgba(0, 0, 0, 0.04)'
//               }
//             }}
//           >
//             대시보드로 돌아가기
//           </Button>
          
//           {/* 다른 분석 유형으로 이동하는 버튼 */}
//           <Button
//             variant="outlined"
//             onClick={() => navigate('/analysis/script')}
//             sx={{ 
//               borderRadius: 8,
//               px: 3,
//               py: 1,
//               borderColor: '#000',
//               color: '#000',
//               '&:hover': {
//                 borderColor: '#333',
//                 backgroundColor: 'rgba(0, 0, 0, 0.04)'
//               }
//             }}
//           >
//             대본 분석 보기
//           </Button>
//         </Box>
        
//         {/* 푸터 */}
//         <Box 
//           component="footer"
//           sx={{ 
//             mt: 4, 
//             textAlign: 'center', 
//             borderTop: '1px solid #eee',
//             pt: 3,
//             pb: 2,
//             color: '#666'
//           }}
//         >
//           <Typography variant="body2">
//             © 2025 PRESENT INSIGHT. All rights reserved.
//           </Typography>
//         </Box>
//       </Container>
//     </Box>
//   );
// };

// export default NonverbalEvaluationDetailPage;

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
import { useNavigate, useParams } from 'react-router-dom';
import {
  ArrowBack as ArrowBackIcon,
  Info as InfoIcon,
  VideoLibrary as VideoLibraryIcon,
  Category as CategoryIcon,
  ChevronLeft as ChevronLeftIcon,
  ChevronRight as ChevronRightIcon,
} from '@mui/icons-material';
import axios from 'axios';

interface NonverbalSegment {
  sample_number: number;
  time_range: string;
  is_normal: boolean;
  top_classes: Array<{ class: string; probability: number }>;
}

interface PresentationAnalysis {
  _id: string;
  filename: string;
  nonverbal_analysis: NonverbalSegment[];
  title: string;
}

interface CategoryCount {
  category: string;
  count: number;
}

const BEHAVIOR_CATEGORIES: { [key: string]: string } = {
  '자세': '#63e6be',
  '손동작': '#ff6b6b',
  '머리동작': '#4dabf7',
  '팔동작': '#4dabf7'
};

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
  const { presentationId } = useParams<{ presentationId?: string }>();

  const [presentation, setPresentation] = useState<PresentationAnalysis | null>(null);
  const [analysisData, setAnalysisData] = useState<NonverbalSegment[] | null>(null);
  const [categoryCountData, setCategoryCountData] = useState<CategoryCount[]>([]);
  const [selectedSegmentIndex, setSelectedSegmentIndex] = useState<number | null>(null);
  const [selectedBehavior, setSelectedBehavior] = useState<string>("");
  const [selectedFeedback, setSelectedFeedback] = useState<string>("구간을 선택하면 상세 피드백이 표시됩니다.");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showContent, setShowContent] = useState(false);
  const [showFeedback, setShowFeedback] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setShowContent(true), 300);
    return () => clearTimeout(timer);
  }, []);

  const scrollTimeline = (direction: 'left' | 'right') => {
    if (timelineRef.current) {
      const scrollAmount = 200;
      const currentScroll = timelineRef.current.scrollLeft;
      timelineRef.current.scrollTo({
        left: direction === 'left' ? currentScroll - scrollAmount : currentScroll + scrollAmount,
        behavior: 'smooth'
      });
    }
  };

  useEffect(() => {
    const fetchNonverbalData = async () => {
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
          nonverbal_analysis: data.nonverbal_analysis,
          title: data.filename.split('.')[0],
        };

        setPresentation(presentationData);
        setAnalysisData(presentationData.nonverbal_analysis);
        processAnalysisData(presentationData.nonverbal_analysis);
      } catch (err: any) {
        console.error("발표 데이터 로드 에러:", err);
        setError(err.response?.data?.error || '데이터를 불러오는 중 오류가 발생했습니다.');
      } finally {
        setLoading(false);
      }
    };

    fetchNonverbalData();
  }, [presentationId]);

  const processAnalysisData = useCallback((data: NonverbalSegment[]) => {
    if (!data || data.length === 0) return;

    const categoryCounts: Record<string, number> = {
      '손동작': 0,
      '머리동작': 0,
      '팔동작': 0,
      '자세': 0
    };

    let firstAbnormalSegmentIndex: number | null = null;

    data.forEach((segment, index) => {
      if (!segment.is_normal && segment.top_classes && segment.top_classes.length > 0) {
        const topClass = segment.top_classes[0].class;

        if (firstAbnormalSegmentIndex === null) {
          firstAbnormalSegmentIndex = index;
        }

        for (const category of Object.keys(categoryCounts)) {
          if (topClass.startsWith(category)) {
            categoryCounts[category] += 1;
            break;
          }
        }
      }
    });

    const chartData: CategoryCount[] = Object.entries(categoryCounts).map(([category, count]) => ({
      category,
      count
    }));

    setCategoryCountData(chartData);

    if (firstAbnormalSegmentIndex !== null) {
      handleSegmentSelect(firstAbnormalSegmentIndex);
    }
  }, []);

  const handleSegmentSelect = (index: number) => {
    if (!analysisData || !analysisData[index]) return;

    const segment = analysisData[index];
    setSelectedSegmentIndex(index);

    if (segment.is_normal) {
      setSelectedBehavior("정상");
      setSelectedFeedback("이 구간에서는 특별한 문제가 감지되지 않았습니다. 좋은 자세와 제스처를 유지하고 있습니다.");
    } else if (segment.top_classes && segment.top_classes.length > 0) {
      const topBehavior = segment.top_classes[0].class;
      setSelectedBehavior(topBehavior);
      const feedback = BEHAVIOR_FEEDBACK[topBehavior] || "이 행동에 대한 구체적인 피드백이 없습니다.";
      setSelectedFeedback(feedback);
    }

    setShowFeedback(false);
    setTimeout(() => {
      setShowFeedback(true);
    }, 300);
  };

  const getCategoryColor = (category: string): string => {
    return BEHAVIOR_CATEGORIES[category] || '#888888';
  };

  const handleNavigateBack = () => {
    if (presentationId) {
      navigate(`/analysis/${presentationId}`);
    } else {
      navigate('/analysis');
    }
  };

  const getSegmentThumbnail = (segment: NonverbalSegment): string => {
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
              {presentation && ` - ${presentation.title}`}
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
                      <Box 
                        sx={{ 
                          width: '100%', 
                          pb: '56.25%',
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
                      
                      <Typography variant="subtitle2" fontWeight={600} sx={{ mb: 1 }}>
                        타임라인
                      </Typography>

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
                                width: '100px',
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
            
            <Grid item xs={12} md={6}>
              <Slide direction="left" in={showContent} timeout={800}>
                <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column', gap: 3 }}>
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
                            {!analysisData[selectedSegmentIndex].is_normal && analysisData[selectedSegmentIndex].top_classes && (
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