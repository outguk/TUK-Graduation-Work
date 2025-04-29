import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Typography,
  Container,
  CircularProgress,
  Alert,
  Paper,
  Fade,
  Grid,
  Card,
  CardContent,
  Tooltip,
  IconButton,
} from '@mui/material';
import { 
  CloudUpload as CloudUploadIcon, 
  ArrowBack as ArrowBackIcon,
  InfoOutlined as InfoOutlinedIcon
} from '@mui/icons-material';
import axios from 'axios';
import guideImg from '../assets/guide.png';
import dashBoard from '../assets/dashboard.png';
import evaluationDetail from '../assets/evaluation_detail.png';

import { useNavigate } from 'react-router-dom';

const UploadPage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showContent, setShowContent] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const navigate = useNavigate();

  // Animation effect on mount
  useEffect(() => {
    const timer = setTimeout(() => setShowContent(true), 300);
    return () => clearTimeout(timer);
  }, []);

  const handleUpload = async () => {
    if (!file) {
      setError('파일을 선택해주세요.');
      return;
    }

    const token = localStorage.getItem('token');
    if (!token) {
      setError('로그인이 필요합니다.');
      navigate('/login');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    setLoading(true);
    setError(null);

    try {
      const response = await axios.post('/spring/api/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          Authorization: `Bearer ${token}`,
        },
      });
      if (response.data.message !== '파일 업로드 및 분석 성공!') {
        throw new Error(response.data.message || '업로드 실패');
      }
      navigate(`/analysis/${response.data.filename}`);
    } catch (e: any) {
      const errorMessage = e.response?.data?.message || '업로드 또는 분석 중 오류가 발생했습니다.';
      setError(errorMessage);
      console.error('업로드 오류:', e);
    } finally {
      setLoading(false);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  return (
    <Box
      component="main"
      sx={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        py: { xs: 6, md: 8 },
        background: '#FFFFFF',
      }}
    >
      <Container maxWidth="lg">
        {/* 로고 및 브랜드 요소 */}
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'center',
            mb: 5
          }}
        >
          <Typography 
            variant="h6" 
            fontWeight={700}
            sx={{
              letterSpacing: 1.2,
              color: '#000',
              p: 1,
              borderBottom: '2px solid #000'
            }}
          >
            PRESENT INSIGHT
          </Typography>
        </Box>

        <Fade in={showContent} timeout={1000}>
          <Box>
            {/* 헤더 영역 - 제목과 뒤로가기 버튼 (같은 높이에 배치) */}
            <Box sx={{ 
              display: 'flex', 
              justifyContent: 'center',
              alignItems: 'center',
              position: 'relative',
              mb: 4
            }}>
              {/* 메인으로 돌아가기 버튼 - 왼쪽 배치 (업로드 박스의 왼쪽 끝과 일치) */}
              <Box sx={{ 
                position: 'absolute', 
                left: 0, 
                top: '55%', // 정가운데(50%)보다 살짝 아래로
                transform: 'translateY(-50%)', // 버튼 자체 높이의 절반만큼 위로 조정
                display: 'flex', 
                alignItems: 'center', 
                height: '100%' 
              }}>
                <Button
                  variant="outlined"
                  startIcon={<ArrowBackIcon />}
                  onClick={() => navigate('/main')}
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
                  aria-label="메인으로 돌아가기"
                >
                  메인으로 돌아가기
                </Button>
              </Box>
              
              {/* 헤드라인 - 중앙 배치 */}
              <Typography 
                variant="h4" 
                component="h1"
                fontWeight={700}
                align="center"
                sx={{
                  color: '#000',
                  position: 'relative',
                  fontSize: { xs: '1.8rem', md: '2.2rem' }
                }}
              >
                발표 영상 업로드
              </Typography>
            </Box>

            {/* 메인 콘텐츠 영역 */}
            <Grid container spacing={3} sx={{ mb: 4 }}>
              {/* 왼쪽 컬럼: 상단 업로드 영역 + 하단 가이드 영역 */}
              <Grid item xs={12} md={6} sx={{ display: 'flex', flexDirection: 'column' }}>
                {/* 업로드 영역 */}
                <Paper
                  elevation={0}
                  sx={{
                    p: { xs: 2, md: 3 },
                    borderRadius: 4,
                    boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
                    border: dragActive ? '2px dashed #000' : '1px solid #eaeaea',
                    position: 'relative',
                    overflow: 'hidden',
                    transition: 'all 0.3s ease',
                    mb: 3, // 아래 분석 과정 가이드와의 간격
                    '&:hover': {
                      boxShadow: '0 6px 25px rgba(0,0,0,0.12)',
                    },
                    '&::after': {
                      content: '""',
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      width: '100%',
                      height: '4px',
                      background: '#000',
                      opacity: 0,
                      transition: 'opacity 0.3s ease'
                    },
                    '&:hover::after': {
                      opacity: 1
                    }
                  }}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                >
                  <Box 
                    sx={{ 
                      display: 'flex', 
                      flexDirection: 'column', 
                      alignItems: 'center', 
                      justifyContent: 'center',
                      py: { xs: 1, md: 1.5 } // 세로 패딩 감소
                    }}
                  >
                    {/* 업로드 아이콘 */}
                    <Box 
                      sx={{ 
                        display: 'flex',
                        justifyContent: 'center',
                        alignItems: 'center',
                        width: '50px', // 아이콘 크기 감소
                        height: '50px', // 아이콘 크기 감소
                        borderRadius: '50%',
                        background: 'rgba(0, 0, 0, 0.05)',
                        mb: 1
                      }}
                    >
                      <CloudUploadIcon sx={{ fontSize: 24, color: '#000' }} />
                    </Box>

                    {/* 업로드 안내 텍스트 */}
                    <Typography 
                      variant="h6" 
                      align="center" 
                      gutterBottom
                      fontWeight={600}
                      sx={{ mb: 0.5, fontSize: '1.1rem' }} // 글자 크기 감소
                    >
                      이곳에 영상을 올려주세요
                    </Typography>

                    <Typography 
                      variant="body2" 
                      align="center" 
                      color="text.secondary"
                      sx={{ mb: 1.5, maxWidth: 500 }}
                    >
                      MP4 형식의 발표 영상을 업로드하고 AI 분석을 통해 개선점을 확인하세요.
                    </Typography>

                    {/* 파일 선택 영역 */}
                    <input
                      id="video-upload"
                      type="file"
                      accept="video/mp4"
                      onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                      style={{ display: 'none' }}
                    />

                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1.5 }}>
                      <Button
                        component="label"
                        htmlFor="video-upload"
                        variant="outlined"
                        size="small" // 버튼 크기 감소
                        sx={{ 
                          borderRadius: 2,
                          px: 2,
                          py: 0.5,
                          mr: 2,
                          borderColor: '#000',
                          color: '#000',
                          '&:hover': {
                            borderColor: '#333',
                            backgroundColor: 'rgba(0, 0, 0, 0.04)'
                          }
                        }}
                      >
                        파일 선택
                      </Button>

                      <Typography variant="body2" color="text.secondary">
                        {file ? file.name : '선택된 파일 없음'}
                      </Typography>
                    </Box>

                    {/* 최대 파일 크기 안내 */}
                    <Typography 
                      variant="caption" 
                      color="text.secondary" 
                      sx={{ mb: 1.5 }}
                    >
                      최대 300MB까지 가능합니다
                    </Typography>

                    {/* 업로드 버튼 */}
                    <Button
                      variant="contained"
                      onClick={handleUpload}
                      disabled={!file || loading}
                      size="medium" // 버튼 크기 감소
                      sx={{ 
                        px: 3, 
                        py: 1, 
                        borderRadius: 10, 
                        fontWeight: 600, 
                        fontSize: '0.9rem',
                        backgroundColor: '#000',
                        color: '#fff',
                        boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
                        transition: 'all 0.3s ease',
                        '&:hover': {
                          boxShadow: '0 6px 15px rgba(0, 0, 0, 0.25)',
                          transform: 'translateY(-3px)',
                          backgroundColor: '#333'
                        }
                      }}
                    >
                      {loading ? (
                        <CircularProgress size={20} color="inherit" sx={{ mr: 1 }} />
                      ) : null}
                      {loading ? '분석 중...' : '업로드 및 분석 시작'}
                    </Button>
                  </Box>
                </Paper>
                
                {/* 분석 과정 가이드 - 업로드 박스 아래에 위치 */}
                <Box sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
                  <Typography 
                    variant="subtitle1" 
                    fontWeight={600}
                    sx={{ mb: 2, pl: 1 }}
                  >
                    분석 과정 가이드
                  </Typography>
                  
                  <Grid container spacing={2} sx={{ flexGrow: 1 }}>
                    {/* STEP 1 카드 */}
                    <Grid item xs={12} sm={6} sx={{ display: 'flex' }}>
                      <Card 
                        sx={{ 
                          borderRadius: 3, 
                          boxShadow: '0 4px 8px rgba(0,0,0,0.08)',
                          width: '100%',
                          transition: 'all 0.3s ease-in-out',
                          '&:hover': {
                            transform: 'translateY(-8px)',
                            boxShadow: '0 14px 28px rgba(0,0,0,0.15)'
                          },
                          overflow: 'hidden',
                          position: 'relative',
                          '&::after': {
                            content: '""',
                            position: 'absolute',
                            top: 0,
                            left: 0,
                            width: '100%',
                            height: '4px',
                            background: '#000',
                            opacity: 0,
                            transition: 'opacity 0.3s ease'
                          },
                          '&:hover::after': {
                            opacity: 1
                          }
                        }}
                      >
                        <CardContent sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                            <Typography 
                              variant="subtitle1" 
                              component="h3"
                              fontWeight={600}
                            >
                              STEP 1
                            </Typography>
                            <Tooltip title="왼쪽 사이드바에서 지난 분석 결과를 확인할 수 있습니다">
                              <IconButton size="small">
                                <InfoOutlinedIcon fontSize="small" />
                              </IconButton>
                            </Tooltip>
                          </Box>
                          
                          <Typography 
                            variant="body2" 
                            fontWeight={500}
                            sx={{ mb: 1 }}
                          >
                            발표 전체 요약 정보 확인
                          </Typography>
                          
                          <Box 
                            sx={{ 
                              width: '100%', 
                              pt: '56.25%',
                              position: 'relative',
                              borderRadius: 2,
                              overflow: 'hidden',
                              mb: 1,
                              border: '1px solid #eee',
                              flexGrow: 1
                            }}
                          >
                            <Box 
                              component="img"
                              src={dashBoard}
                              alt="분석 대시보드 예시"
                              sx={{
                                position: 'absolute',
                                top: 0,
                                left: 0,
                                width: '100%',
                                height: '100%',
                                objectFit: 'cover'
                              }}
                            />
                          </Box>
                          
                          <Typography variant="caption" color="text.secondary">
                            분석이 완료되면 발표 속도, 음량, 비언어적 요소 등 전반적인 분석 결과를 확인할 수 있습니다.
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                    
                    {/* STEP 2 카드 */}
                    <Grid item xs={12} sm={6} sx={{ display: 'flex' }}>
                      <Card 
                        sx={{ 
                          borderRadius: 3, 
                          boxShadow: '0 4px 8px rgba(0,0,0,0.08)',
                          width: '100%',
                          transition: 'all 0.3s ease-in-out',
                          '&:hover': {
                            transform: 'translateY(-8px)',
                            boxShadow: '0 14px 28px rgba(0,0,0,0.15)'
                          },
                          overflow: 'hidden',
                          position: 'relative',
                          '&::after': {
                            content: '""',
                            position: 'absolute',
                            top: 0,
                            left: 0,
                            width: '100%',
                            height: '4px',
                            background: '#000',
                            opacity: 0,
                            transition: 'opacity 0.3s ease'
                          },
                          '&:hover::after': {
                            opacity: 1
                          }
                        }}
                      >
                        <CardContent sx={{ p: 2, height: '100%', display: 'flex', flexDirection: 'column' }}>
                          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                            <Typography 
                              variant="subtitle1"
                              component="h3"
                              fontWeight={600}
                            >
                              STEP 2
                            </Typography>
                            <Tooltip title="각 평가 항목별 상세 분석 결과를 확인합니다">
                              <IconButton size="small">
                                <InfoOutlinedIcon fontSize="small" />
                              </IconButton>
                            </Tooltip>
                          </Box>
                          
                          <Typography 
                            variant="body2" 
                            fontWeight={500}
                            sx={{ mb: 1 }}
                          >
                            요소별 세부 평가 결과 확인
                          </Typography>
                          
                          <Box 
                            sx={{ 
                              width: '100%', 
                              pt: '56.25%',
                              position: 'relative',
                              borderRadius: 2,
                              overflow: 'hidden',
                              mb: 1,
                              border: '1px solid #eee',
                              flexGrow: 1
                            }}
                          >
                            <Box 
                              component="img"
                              src={evaluationDetail}
                              alt="세부 평가 화면 예시"
                              sx={{
                                position: 'absolute',
                                top: 0,
                                left: 0,
                                width: '100%',
                                height: '100%',
                                objectFit: 'cover'
                              }}
                            />
                            
                            {/* 세부 평가 화면의 차트 영역 강조 표시 */}
                            {/*<Box*/}
                            {/*  sx={{*/}
                            {/*    position: 'absolute',*/}
                            {/*    top: '30%',*/}
                            {/*    left: '20%',*/}
                            {/*    width: '40%',*/}
                            {/*    height: '30%',*/}
                            {/*    border: '2px solid #ff0000',*/}
                            {/*    borderRadius: 1,*/}
                            {/*    zIndex: 2*/}
                            {/*  }}*/}
                            {/*/>*/}
                          </Box>
                          
                          <Typography variant="caption" color="text.secondary">
                            발표의 각 항목별 세부 분석 결과와 개선 피드백을 받을 수 있습니다.
                          </Typography>
                        </CardContent>
                      </Card>
                    </Grid>
                  </Grid>
                </Box>
              </Grid>
              
              {/* 오른쪽 컬럼: 주의사항 영역 */}
              <Grid item xs={12} md={6}>
                <Paper
                  elevation={0}
                  sx={{
                    p: { xs: 2, md: 3 },
                    borderRadius: 4,
                    boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
                    border: '1px solid #eaeaea',
                    position: 'relative',
                    overflow: 'hidden',
                    transition: 'all 0.3s ease',
                    height: '100%', // 좌측 업로드 박스 + 분석 가이드 높이와 맞춤
                    '&:hover': {
                      boxShadow: '0 6px 25px rgba(0,0,0,0.12)',
                      transform: 'translateY(-4px)', // 호버 시 위로 약간 이동하는 애니메이션 추가
                    },
                    '&::after': {
                      content: '""',
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      width: '100%',
                      height: '4px',
                      background: '#000',
                      opacity: 0,
                      transition: 'opacity 0.3s ease'
                    },
                    '&:hover::after': {
                      opacity: 1
                    },
                    display: 'flex',
                    flexDirection: 'column'
                  }}
                >
                  <Typography 
                    variant="h6" 
                    fontWeight={600}
                    sx={{ mb: 2 }}
                  >
                    영상 업로드 주의사항
                  </Typography>
                  
                  {/* 이미지 영역 */}
                  <Box 
                    sx={{ 
                      width: '100%', 
                      pt: '40%', // 높이 비율 감소
                      position: 'relative',
                      borderRadius: 2,
                      overflow: 'hidden',
                      mb: 2,
                      backgroundColor: '#f8f8f8'
                    }}
                  >
                    <Box 
                      component="img"
                      src={guideImg}
                      alt="발표 영상 가이드 이미지"
                      sx={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%',
                        objectFit: 'contain'
                      }}
                    />
                  </Box>
                  
                  {/* 주의사항 내용 영역 (텍스트 크기 감소) */}
                  <Box sx={{ px: 1, flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
                    <Typography variant="subtitle2" fontWeight={600} sx={{ mb: 0.5 }}>
                      업로드 전 확인사항
                    </Typography>
                    
                    <Box component="ul" sx={{ pl: 2, mb: 1.5, mt: 0.5 }}>
                      <Typography component="li" variant="body2" sx={{ mb: 0.5 }}>
                        <b>파일 형식:</b> MP4 형식의 영상만 업로드 가능합니다.
                      </Typography>
                      <Typography component="li" variant="body2" sx={{ mb: 0.5 }}>
                        <b>최대 용량:</b> 300MB 이하의 영상을 권장합니다.
                      </Typography>
                      <Typography component="li" variant="body2" sx={{ mb: 0.5 }}>
                        <b>영상 길이:</b> 4분 이내의 발표 영상이 최적의 분석 결과를 제공합니다.
                      </Typography>
                    </Box>
                    
                    <Typography variant="subtitle2" fontWeight={600} sx={{ mb: 0.5 }}>
                      최적의 영상 촬영 조건:
                    </Typography>
                    
                    <Box component="ul" sx={{ pl: 2, mt: 0.5 }}>
                      <Typography component="li" variant="body2" sx={{ mb: 0.5 }}>
                        <b>조명:</b> 발표자의 얼굴과 몸이 잘 보이는 밝은 환경에서 촬영하세요.
                      </Typography>
                      <Typography component="li" variant="body2" sx={{ mb: 0.5 }}>
                        <b>거리:</b> 발표자의 전신이 잘 보이도록 적절한 거리를 유지하세요.
                      </Typography>
                      <Typography component="li" variant="body2" sx={{ mb: 0.5 }}>
                        <b>소리:</b> 주변 소음이 적은 환경에서 마이크를 사용하여 녹음하세요.
                      </Typography>
                    </Box>
                    
                    {/* 추가 팁 */}
                    <Box sx={{ mt: 'auto', bgcolor: '#f5f5f5', p: 2, borderRadius: 2 }}>
                      <Typography variant="subtitle2" fontWeight={600} sx={{ mb: 1 }}>
                        💡 프로 팁
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        영상을 업로드하기 전에 미리보기로 확인하여 화면과 소리가 잘 녹화되었는지 검토하세요.
                        최적의 분석 결과를 위해 정면을 바라보고 선명한 발표 모습을 촬영하세요.
                      </Typography>
                    </Box>
                  </Box>
                </Paper>
              </Grid>
            </Grid>

            {/* 에러 메시지 */}
            {error && (
              <Alert 
                severity="error" 
                sx={{ mb: 4 }} 
                onClose={() => setError(null)}
              >
                {error}
              </Alert>
            )}
          </Box>
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
            © 2025 PRESENT INSIGHT. All rights reserved.
          </Typography>
        </Box>
      </Container>
    </Box>
  );
};

export default UploadPage;