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

import React, { useState, useEffect, useCallback, useRef } from "react";
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
} from "@mui/material";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { useNavigate, useParams } from "react-router-dom";
import {
  ArrowBack as ArrowBackIcon,
  Info as InfoIcon,
  VideoLibrary as VideoLibraryIcon,
  Category as CategoryIcon,
  ChevronLeft as ChevronLeftIcon,
  ChevronRight as ChevronRightIcon,
} from "@mui/icons-material";
import axios from "axios";

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
  자세: "#63e6be",
  손동작: "#ff6b6b",
  머리동작: "#4dabf7",
  팔동작: "#ffd400",
};

const BEHAVIOR_FEEDBACK: { [key: string]: string } = {
  "손동작(머리)":
    "머리를 만지는 습관은 불안한 인상을 줄 수 있어요. 손은 안정된 위치에 두는 것이 좋아요.",
  "손동작(얼굴)":
    "얼굴을 만지는 행동은 산만해 보일 수 있어요. 청중과의 시선 유지에 집중해보세요.",
  "손동작(몸긁기)":
    "몸을 긁는 동작은 불편함을 드러낼 수 있어요. 손은 가볍게 모으거나 제스처로 활용해보세요.",
  "손동작(손톱)":
    "손톱을 만지거나 뜯는 동작은 긴장감을 전달해요. 손을 자연스럽게 두는 연습이 필요해요.",
  "머리동작(고개흔들기)":
    "고개를 자주 흔드는 것은 산만한 인상을 줄 수 있어요. 시선과 자세를 고정해보세요.",
  "머리동작(좌우흔들기)":
    "머리를 좌우로 흔드는 습관은 주의를 흩뜨릴 수 있어요. 차분한 고개 움직임을 유지하세요.",
  "머리동작(숙이기)":
    "고개를 숙이는 자세는 자신감 부족으로 보일 수 있어요. 시선을 정면으로 유지해보세요.",
  "팔동작(뒷짐)":
    "뒷짐은 청중과의 거리감을 줄 수 있어요. 앞에 두고 자연스러운 제스처를 사용하는 것이 좋습니다.",
  "팔동작(무의미반동)":
    "불필요한 팔의 움직임은 발표 흐름을 방해할 수 있어요. 의미 있는 제스처만 사용하는 연습이 필요해요.",
  "자세(좌우흔들기)":
    "몸을 좌우로 흔드는 습관은 발표자의 긴장을 드러냅니다. 중심을 잡고 안정된 자세를 유지해보세요.",
  "자세(비스듬히)":
    "비스듬한 자세는 집중이 부족해 보일 수 있어요. 정면을 향한 단단한 자세가 신뢰감을 줍니다.",
  "자세(비비꼬기)":
    "다리를 꼬는 자세는 불안정하고 산만한 인상을 줄 수 있어요. 두 다리를 안정적으로 두는 자세를 연습하세요.",
};

const NonverbalEvaluationDetailPage: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const FASTAPI_BASE = "http://localhost:5000";

  const navigate = useNavigate();
  const timelineRef = useRef<HTMLDivElement>(null);
  const { presentationId } = useParams<{ presentationId?: string }>();

  const [presentation, setPresentation] = useState<PresentationAnalysis | null>(
    null
  );
  const [analysisData, setAnalysisData] = useState<NonverbalSegment[] | null>(
    null
  );
  const [categoryCountData, setCategoryCountData] = useState<CategoryCount[]>(
    []
  );
  const [selectedSegmentIndex, setSelectedSegmentIndex] = useState<
    number | null
  >(null);
  const [selectedBehavior, setSelectedBehavior] = useState<string>("");
  const [selectedFeedback, setSelectedFeedback] = useState<string>(
    "구간을 선택하면 상세 피드백이 표시됩니다."
  );
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showContent, setShowContent] = useState(false);
  const [showFeedback, setShowFeedback] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setShowContent(true), 300);
    return () => clearTimeout(timer);
  }, []);

  const scrollTimeline = (direction: "left" | "right") => {
    if (timelineRef.current) {
      const scrollAmount = 200;
      const currentScroll = timelineRef.current.scrollLeft;
      timelineRef.current.scrollTo({
        left:
          direction === "left"
            ? currentScroll - scrollAmount
            : currentScroll + scrollAmount,
        behavior: "smooth",
      });
    }
  };

 // "22.00s" 같은 문자열을 초 단위 숫자로 변환
 const parseSeconds = (timeStr: string) =>
   parseFloat(timeStr.replace(/[^\d.]/g, ""));


  // 세그먼트로 시킹
 const seekToSegment = (index: number) => {
   if (!analysisData || !videoRef.current) return;
   // "22.00s ~ 23.80s" 에서 앞쪽만 꺼내기
   const range = analysisData[index].time_range;
   const startPart = range.split("~")[0].trim();  // "22.00s"
   const seconds = parseSeconds(startPart);
   videoRef.current.currentTime = seconds;
   videoRef.current.play();
 };

  useEffect(() => {
    const fetchNonverbalData = async () => {
      setLoading(true);
      setError(null);

      try {
        const token = localStorage.getItem("token");
        if (!token) {
          throw new Error("인증 토큰이 없습니다.");
        }

        let response;
        if (presentationId) {
          response = await axios.get(`/spring/api/get-analysis`, {
            params: { filename: presentationId },
            headers: { Authorization: `Bearer ${token}` },
          });
        } else {
          response = await axios.get(`/spring/api/my-analyses`, {
            headers: { Authorization: `Bearer ${token}` },
          });
          if (response.data.analyses && response.data.analyses.length > 0) {
            response.data = response.data.analyses[0];
          } else {
            throw new Error("분석 데이터가 없습니다.");
          }
        }

        const data = response.data;
        const presentationData: PresentationAnalysis = {
          _id: data._id,
          filename: data.filename,
          nonverbal_analysis: data.nonverbal_analysis,
          title: data.filename.split(".")[0],
        };

        setPresentation(presentationData);
        setAnalysisData(presentationData.nonverbal_analysis);
        processAnalysisData(presentationData.nonverbal_analysis);
      } catch (err: any) {
        console.error("발표 데이터 로드 에러:", err);
        setError(
          err.response?.data?.error ||
            "데이터를 불러오는 중 오류가 발생했습니다."
        );
      } finally {
        setLoading(false);
      }
    };

    fetchNonverbalData();
  }, [presentationId]);

  const processAnalysisData = useCallback((data: NonverbalSegment[]) => {
    if (!data || data.length === 0) return;

    const categoryCounts: Record<string, number> = {
      손동작: 0,
      머리동작: 0,
      팔동작: 0,
      자세: 0,
    };

    let firstAbnormalSegmentIndex: number | null = null;

    data.forEach((segment, index) => {
      if (
        !segment.is_normal &&
        segment.top_classes &&
        segment.top_classes.length > 0
      ) {
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

    const chartData: CategoryCount[] = Object.entries(categoryCounts).map(
      ([category, count]) => ({
        category,
        count,
      })
    );

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
      setSelectedFeedback(
        "이 구간에서는 특별한 문제가 감지되지 않았습니다. 좋은 자세와 제스처를 유지하고 있습니다."
      );
    } else if (segment.top_classes && segment.top_classes.length > 0) {
      const topBehavior = segment.top_classes[0].class;
      setSelectedBehavior(topBehavior);
      const feedback =
        BEHAVIOR_FEEDBACK[topBehavior] ||
        "이 행동에 대한 구체적인 피드백이 없습니다.";
      setSelectedFeedback(feedback);
    }

    setShowFeedback(false);
    setTimeout(() => {
      setShowFeedback(true);
    }, 300);

    // 영상도 해당 구간으로 이동시키기
    seekToSegment(index);
  };

  const getCategoryColor = (category: string): string => {
    return BEHAVIOR_CATEGORIES[category] || "#888888";
  };

  const handleNavigateBack = () => {
    if (presentationId) {
      navigate(`/analysis/${presentationId}`);
    } else {
      navigate("/analysis");
    }
  };

  // const getSegmentThumbnail = (segment: NonverbalSegment): string => {
  //   return `/api/placeholder/640/360?text=구간 ${segment.sample_number} (${segment.time_range})`;
  // };

  return (
    <Box
      component="main"
      sx={{
        minHeight: "100vh",
        py: { xs: 4, md: 6 },
        px: { xs: 2, md: 4 },
        background: "#FFFFFF",
      }}
    >
      <Container maxWidth="xl">
        <Box sx={{ mb: 5, display: "flex", alignItems: "center" }}>
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
                position: "relative",
                display: "inline-block",
              }}
            >
              비언어적 분석
              {presentation && ` - ${presentation.title}`}
            </Typography>
            <Typography variant="body1" color="text.secondary" sx={{ mt: 1 }}>
              발표 중 자세, 손동작, 머리 움직임 등의 비언어적 요소 분석
              결과입니다.
            </Typography>
          </Box>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
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
                    boxShadow: "0 4px 20px rgba(0,0,0,0.08)",
                    height: "100%",
                    display: "flex",
                    flexDirection: "column",
                    position: "relative",
                    overflow: "hidden",
                    "&::before": {
                      content: '""',
                      position: "absolute",
                      top: 0,
                      left: 0,
                      width: "100%",
                      height: "4px",
                      background: "#000",
                    },
                  }}
                >
                  <Box
                    sx={{
                      display: "flex",
                      justifyContent: "space-between",
                      alignItems: "center",
                      mb: 2,
                    }}
                  >
                    <Typography variant="h6" component="h2" fontWeight={600}>
                      발표 영상 구간별 분석
                    </Typography>
                    <VideoLibraryIcon />
                  </Box>

                  <Typography
                    variant="body2"
                    color="text.secondary"
                    sx={{ mb: 3 }}
                  >
                    타임라인에서 특정 구간을 선택하여 상세한 비언어적 분석
                    결과를 확인하세요.
                  </Typography>

                  {loading ? (
                    <Box
                      sx={{
                        flexGrow: 1,
                        display: "flex",
                        justifyContent: "center",
                        alignItems: "center",
                      }}
                    >
                      <CircularProgress size={40} />
                    </Box>
                  ) : (
                    <Box
                      sx={{
                        display: "flex",
                        flexDirection: "column",
                        height: "100%",
                      }}
                    >
                      <Box
                        sx={{
                          width: "100%",
                          pb: "10%",
                          position: "relative",
                          bgcolor: "#f5f5f5",
                          borderRadius: 2,
                          mb: 3,
                          overflow: "hidden",
                          flexGrow: 0,
                        }}
                      >
                        {analysisData && (
                          <Box
                            sx={{
                              width: "100%",
                              pb: "56.25%",
                              position: "relative",
                              bgcolor: "#000",
                              borderRadius: 2,
                              mb: 3,
                              overflow: "hidden",
                              flexGrow: 0,
                            }}
                          >
                            <video
                              ref={videoRef}
                              controls
                              crossOrigin="anonymous"
                              src={`${FASTAPI_BASE}/fastapi/api/video/${
                                presentation!.filename
                              }`}
                              style={{
                                position: "absolute",
                                top: 0,
                                left: 0,
                                width: "100%",
                                height: "100%",
                                objectFit: "cover",
                              }}
                              onLoadedMetadata={() => {
                                // 영상 메타데이터 로드 시 필요하면 초기 세그먼트로 이동
                                if (selectedSegmentIndex !== null) {
                                  seekToSegment(selectedSegmentIndex);
                                }
                              }}
                            />
                          </Box>
                        )}

                        {selectedSegmentIndex !== null && analysisData && (
                          <Box
                            sx={{
                              position: "absolute",
                              bottom: 0,
                              left: 0,
                              width: "100%",
                              bgcolor: "rgba(0, 0, 0, 0.7)",
                              color: "white",
                              p: 2,
                            }}
                          >
                            <Typography variant="body2" fontWeight={500}>
                              구간:{" "}
                              {analysisData[selectedSegmentIndex].time_range}
                            </Typography>
                            <Typography variant="caption">
                              {analysisData[selectedSegmentIndex].is_normal
                                ? "정상적인 자세와 제스처"
                                : `감지된 행동: ${selectedBehavior}`}
                            </Typography>
                          </Box>
                        )}
                      </Box>

                      <Typography
                        variant="subtitle2"
                        fontWeight={600}
                        sx={{ mb: 1 }}
                      >
                        타임라인
                      </Typography>

                      <Box
                        sx={{
                          display: "flex",
                          alignItems: "center",
                          mb: 1,
                          justifyContent: "space-between",
                        }}
                      >
                        <IconButton
                          size="small"
                          onClick={() => scrollTimeline("left")}
                          aria-label="타임라인 왼쪽으로 이동"
                        >
                          <ChevronLeftIcon />
                        </IconButton>

                        <Typography variant="caption" color="text.secondary">
                          좌우로 스크롤하여 더 많은 구간 확인
                        </Typography>

                        <IconButton
                          size="small"
                          onClick={() => scrollTimeline("right")}
                          aria-label="타임라인 오른쪽으로 이동"
                        >
                          <ChevronRightIcon />
                        </IconButton>
                      </Box>

                      <Box
                        sx={{
                          position: "relative",
                          width: "100%",
                          overflow: "hidden",
                        }}
                      >
                        <Box
                          ref={timelineRef}
                          sx={{
                            display: "flex",
                            overflowX: "auto",
                            width: "100%",
                            scrollbarWidth: "thin",
                            "&::-webkit-scrollbar": {
                              height: "6px",
                            },
                            "&::-webkit-scrollbar-track": {
                              backgroundColor: "#f1f1f1",
                              borderRadius: "10px",
                            },
                            "&::-webkit-scrollbar-thumb": {
                              backgroundColor: "#888",
                              borderRadius: "10px",
                            },
                            pb: 1,
                          }}
                        >
                          {analysisData &&
                            analysisData.map((segment, index) => (
                              <Box
                                key={index}
                                onClick={() => handleSegmentSelect(index)}
                                sx={{
                                  width: "100px",
                                  minWidth: "100px",
                                  height: "60px",
                                  bgcolor: segment.is_normal
                                    ? "#4caf50"
                                    : "#ff8c00",
                                  opacity:
                                    selectedSegmentIndex === index ? 1 : 0.7,
                                  cursor: "pointer",
                                  position: "relative",
                                  transition: "all 0.2s ease",
                                  mr: 0.5,
                                  "&:hover": {
                                    opacity: 0.9,
                                    transform: "translateY(-2px)",
                                  },
                                  border:
                                    selectedSegmentIndex === index
                                      ? "2px solid #000"
                                      : "1px solid rgba(255,255,255,0.3)",
                                  borderRadius: 1,
                                  display: "flex",
                                  flexDirection: "column",
                                  justifyContent: "flex-end",
                                  alignItems: "center",
                                }}
                                role="button"
                                aria-label={`발표 구간 ${segment.time_range}`}
                                tabIndex={0}
                              >
                                <Typography
                                  variant="caption"
                                  sx={{
                                    color: "white",
                                    textAlign: "center",
                                    fontSize: "10px",
                                    mb: 0.5,
                                    fontWeight:
                                      selectedSegmentIndex === index
                                        ? "bold"
                                        : "normal",
                                  }}
                                >
                                  {index + 1}
                                </Typography>
                                <Typography
                                  variant="caption"
                                  sx={{
                                    color: "white",
                                    textAlign: "center",
                                    fontSize: "8px",
                                    bgcolor: "rgba(0,0,0,0.3)",
                                    px: 0.5,
                                    py: 0.25,
                                    borderRadius: 0.5,
                                    width: "90%",
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
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          mt: 2,
                          gap: 3,
                        }}
                      >
                        <Box sx={{ display: "flex", alignItems: "center" }}>
                          <Box
                            sx={{
                              width: 16,
                              height: 16,
                              borderRadius: 1,
                              bgcolor: "#4caf50",
                              mr: 1,
                            }}
                          />
                          <Typography variant="caption">정상 구간</Typography>
                        </Box>
                        <Box sx={{ display: "flex", alignItems: "center" }}>
                          <Box
                            sx={{
                              width: 16,
                              height: 16,
                              borderRadius: 1,
                              bgcolor: "#ff8c00",
                              mr: 1,
                            }}
                          />
                          <Typography variant="caption">
                            개선 필요 구간
                          </Typography>
                        </Box>
                      </Box>
                    </Box>
                  )}
                </Paper>
              </Slide>
            </Grid>

            <Grid item xs={12} md={6}>
              <Slide direction="left" in={showContent} timeout={800}>
                <Box
                  sx={{
                    height: "100%",
                    display: "flex",
                    flexDirection: "column",
                    gap: 3,
                  }}
                >
                  <Paper
                    elevation={0}
                    sx={{
                      p: 3,
                      borderRadius: 3,
                      boxShadow: "0 4px 20px rgba(0,0,0,0.08)",
                      position: "relative",
                      overflow: "hidden",
                      flex: "0 0 auto",
                      "&::before": {
                        content: '""',
                        position: "absolute",
                        top: 0,
                        left: 0,
                        width: "100%",
                        height: "4px",
                        background: "#000",
                      },
                    }}
                  >
                    <Box
                      sx={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        mb: 2,
                      }}
                    >
                      <Typography variant="h6" component="h2" fontWeight={600}>
                        비언어적 행동 발생 빈도
                      </Typography>
                      <CategoryIcon />
                    </Box>

                    <Typography
                      variant="body2"
                      color="text.secondary"
                      sx={{ mb: 3 }}
                    >
                      발표 중 감지된 개선이 필요한 비언어적 행동 카테고리별 발생
                      빈도입니다.
                    </Typography>

                    {loading ? (
                      <Box
                        sx={{
                          height: 240,
                          display: "flex",
                          justifyContent: "center",
                          alignItems: "center",
                        }}
                      >
                        <CircularProgress size={40} />
                      </Box>
                    ) : (
                      <Box sx={{ height: 240, width: "100%" }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart
                            data={categoryCountData}
                            margin={{
                              top: 10,
                              right: 10,
                              left: 10,
                              bottom: 20,
                            }}
                          >
                            <CartesianGrid
                              strokeDasharray="3 3"
                              vertical={false}
                            />
                            <XAxis dataKey="category" />
                            <YAxis allowDecimals={false} />
                            <Tooltip
                              formatter={(value) => [`${value}회`, "발생 빈도"]}
                              contentStyle={{
                                backgroundColor: "#fff",
                                borderRadius: "8px",
                                boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
                                border: "none",
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
                      boxShadow: "0 4px 20px rgba(0,0,0,0.08)",
                      position: "relative",
                      overflow: "hidden",
                      flex: "1 1 auto",
                      display: "flex",
                      flexDirection: "column",
                      "&::before": {
                        content: '""',
                        position: "absolute",
                        top: 0,
                        left: 0,
                        width: "100%",
                        height: "4px",
                        background: "#000",
                      },
                    }}
                  >
                    <Typography
                      variant="h6"
                      component="h2"
                      fontWeight={600}
                      sx={{ mb: 3 }}
                    >
                      선택 구간 피드백
                    </Typography>

                    {selectedSegmentIndex !== null && analysisData ? (
                      <Box sx={{ flexGrow: 1 }}>
                        <Box sx={{ mb: 3 }}>
                          <Typography
                            variant="subtitle1"
                            fontWeight={600}
                            gutterBottom
                          >
                            {`선택 구간: ${analysisData[selectedSegmentIndex].time_range}`}
                          </Typography>
                          <Box
                            sx={{
                              display: "flex",
                              alignItems: "center",
                              p: 2,
                              bgcolor: analysisData[selectedSegmentIndex]
                                .is_normal
                                ? "#e8f5e9"
                                : "#fff3e0",
                              borderRadius: 2,
                            }}
                          >
                            <Typography
                              variant="body2"
                              fontWeight={500}
                              color={
                                analysisData[selectedSegmentIndex].is_normal
                                  ? "success.dark"
                                  : "warning.dark"
                              }
                            >
                              상태:{" "}
                              {analysisData[selectedSegmentIndex].is_normal
                                ? "정상"
                                : "개선 필요"}
                            </Typography>
                          </Box>
                        </Box>

                        <Divider sx={{ my: 2 }} />

                        <Fade in={showFeedback} timeout={500}>
                          <Box>
                            {!analysisData[selectedSegmentIndex].is_normal &&
                              analysisData[selectedSegmentIndex]
                                .top_classes && (
                                <Box sx={{ mb: 3 }}>
                                  <Typography
                                    variant="subtitle2"
                                    fontWeight={600}
                                    gutterBottom
                                  >
                                    감지된 행동
                                  </Typography>
                                  <Box
                                    sx={{
                                      p: 2,
                                      borderRadius: 2,
                                      bgcolor: "#f5f5f5",
                                      border: "1px solid #eee",
                                    }}
                                  >
                                    <Typography
                                      variant="body1"
                                      fontWeight={500}
                                    >
                                      {selectedBehavior}
                                    </Typography>
                                    <Typography
                                      variant="caption"
                                      color="text.secondary"
                                    >
                                      확률:{" "}
                                      {
                                        analysisData[selectedSegmentIndex]
                                          .top_classes[0].probability
                                      }
                                      %
                                    </Typography>
                                  </Box>
                                </Box>
                              )}

                            <Typography
                              variant="subtitle2"
                              fontWeight={600}
                              gutterBottom
                            >
                              피드백
                            </Typography>
                            <Paper
                              elevation={0}
                              sx={{
                                p: 3,
                                borderRadius: 3,
                                bgcolor: "#fff",
                                border: "1px solid #eaeaea",
                                boxShadow: "0 2px 10px rgba(0,0,0,0.05)",
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
                          display: "flex",
                          flexDirection: "column",
                          justifyContent: "center",
                          alignItems: "center",
                          flexGrow: 1,
                          p: 3,
                          textAlign: "center",
                        }}
                      >
                        <InfoIcon
                          sx={{ fontSize: 48, color: "text.secondary", mb: 2 }}
                        />
                        <Typography>
                          타임라인에서 구간을 선택하면 해당 구간의 분석 결과가
                          여기에 표시됩니다.
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
            display: "flex",
            justifyContent: "space-between",
            mt: 4,
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
              borderColor: "#000",
              color: "#000",
              "&:hover": {
                borderColor: "#333",
                backgroundColor: "rgba(0, 0, 0, 0.04)",
              },
            }}
          >
            대시보드로 돌아가기
          </Button>

          <Button
            variant="outlined"
            onClick={() => navigate("/analysis/script")}
            sx={{
              borderRadius: 8,
              px: 3,
              py: 1,
              borderColor: "#000",
              color: "#000",
              "&:hover": {
                borderColor: "#333",
                backgroundColor: "rgba(0, 0, 0, 0.04)",
              },
            }}
          >
            대본 분석 보기
          </Button>
        </Box>

        <Box
          component="footer"
          sx={{
            mt: 4,
            textAlign: "center",
            borderTop: "1px solid #eee",
            pt: 3,
            pb: 2,
            color: "#666",
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