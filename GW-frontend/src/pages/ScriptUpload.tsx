import { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
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
  TextField,
} from '@mui/material';
import {
  CloudUpload as CloudUploadIcon,
  ArrowBack as ArrowBackIcon,
} from '@mui/icons-material';
import scriptResultImg from '../assets/scriptresult.png';
import scriptResult2Img from '../assets/scriptresult2.png';

export default function ScriptUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [minutes, setMinutes] = useState<number>(5);
  const navigate = useNavigate();

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showContent, setShowContent] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  const cardStyle = {
    borderRadius: 3,
    boxShadow: '0 4px 8px rgba(0,0,0,0.08)',
    width: '100%',
    transition: 'all 0.3s ease-in-out',
    '&:hover': {
      transform: 'translateY(-8px)',
      boxShadow: '0 14px 28px rgba(0,0,0,0.15)',
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
      transition: 'opacity 0.3s ease',
    },
    '&:hover::after': {
      opacity: 1,
    },
  };

  useEffect(() => {
    const timer = setTimeout(() => setShowContent(true), 300);
    return () => clearTimeout(timer);
  }, []);

  const handleFileSubmit = async () => {
    if (!file) {
      setError('텍스트 파일을 선택해주세요.');
      return;
    }
    if (minutes < 1) {
      setError('발표 시간을 1분 이상 입력하세요.');
      return;
    }

    const token = localStorage.getItem('token');
    if (!token) {
      setError('로그인이 필요합니다.');
      navigate('/login');
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('scriptFile', file);
    formData.append('filename', file.name);
    formData.append('speech_minutes', minutes.toString());

    try {
      const response = await axios.post('/spring/api/script-upload', formData, {
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'multipart/form-data',
        },
      });
      if (response.data && response.data.id) {
        console.log('분석 성공, script_id:', response.data.id);
        navigate(`/script/analysis/${response.data.id}`);
      } else {
        console.error('응답 데이터 구조:', response.data);
        setError('분석 실패: 응답에서 script_id를 찾을 수 없습니다.');
      }
    } catch (err: any) {
      setError('대본 분석 실패: ' + (err.response?.data?.message || err.message));
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
      const droppedFile = e.dataTransfer.files[0];
      if (droppedFile.type === 'text/plain') {
        setFile(droppedFile);
        setError(null);
      } else {
        setError('txt 형식의 파일만 업로드할 수 있습니다.');
      }
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
       if (selectedFile.type === 'text/plain') {
        setFile(selectedFile);
        setError(null);
      } else {
        setError('txt 형식의 파일만 업로드할 수 있습니다.');
        setFile(null);
      }
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
        <Box sx={{ display: 'flex', justifyContent: 'center', mb: 5 }}>
          <Typography
            variant="h6"
            fontWeight={700}
            sx={{
              letterSpacing: 1.2,
              color: '#000',
              p: 1,
              borderBottom: '2px solid #000',
            }}
          >
            PRESENT INSIGHT
          </Typography>
        </Box>

        <Fade in={showContent} timeout={1000}>
          <Box>
            <Box
              sx={{
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                position: 'relative',
                mb: 4,
              }}
            >
              <Box
                sx={{
                  position: 'absolute',
                  left: 0,
                  top: '50%',
                  transform: 'translateY(-50%)',
                }}
              >
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
                      backgroundColor: 'rgba(0, 0, 0, 0.04)',
                    },
                  }}
                  aria-label="메인으로 돌아가기"
                >
                  메인으로 돌아가기
                </Button>
              </Box>

              <Typography
                variant="h4"
                component="h1"
                fontWeight={700}
                align="center"
                sx={{ color: '#000' }}
              >
                대본 업로드
              </Typography>
            </Box>

            <Grid container spacing={4} sx={{ mb: 4 }}>
              <Grid item xs={12} md={8}>
                <Paper
                  elevation={0}
                  sx={{
                    p: { xs: 3, md: 4 },
                    borderRadius: 4,
                    boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
                    border: dragActive ? '2px dashed #000' : '1px solid #eaeaea',
                    position: 'relative',
                    overflow: 'hidden',
                    transition: 'all 0.3s ease',
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
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
                      transition: 'opacity 0.3s ease',
                    },
                    '&:hover::after': {
                      opacity: 1,
                    },
                  }}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                >
                  <Typography variant="h5" component="h2" fontWeight={600} sx={{ mb: 2 }}>
                    대본 파일 업로드
                  </Typography>
                  <CloudUploadIcon sx={{ fontSize: 48, color: 'text.secondary', my: 2 }}/>
                  <Typography variant="body1" align="center" color="text.secondary" sx={{ mb: 3 }}>
                    이곳에 txt 파일을 드래그 앤 드롭하거나<br />파일을 직접 선택하여 업로드하세요.
                  </Typography>
                  
                   <TextField
                      label="예상 발표 시간(분)"
                      type="number"
                      value={minutes}
                      onChange={(e) => setMinutes(Number(e.target.value))}
                      InputProps={{ inputProps: { min: 1 } }}
                      sx={{ mb: 2, width: '80%', maxWidth: '320px' }}
                    />

                  <input
                    id="script-upload-input"
                    type="file"
                    accept=".txt"
                    onChange={handleFileChange}
                    style={{ display: 'none' }}
                  />
                  <Box sx={{ display: 'flex', alignItems: 'center', my: 2 }}>
                     <Button
                        component="label"
                        htmlFor="script-upload-input"
                        variant="outlined"
                        sx={{
                          borderRadius: 2, px: 2, py: 0.5, mr: 2,
                          borderColor: '#000', color: '#000',
                          '&:hover': { borderColor: '#333', backgroundColor: 'rgba(0, 0, 0, 0.04)'}
                        }}
                      >
                        대본 선택
                      </Button>
                      <Typography variant="body2" color="text.secondary" noWrap>
                        {file ? file.name : '선택된 파일 없음'}
                      </Typography>
                  </Box>

                  <Button
                    variant="contained"
                    onClick={handleFileSubmit}
                    disabled={!file || loading || minutes < 1}
                    size="large"
                    sx={{
                      mt: 2, px: 5, py: 1.5, borderRadius: 10,
                      fontWeight: 600, backgroundColor: '#000',
                      color: '#fff', '&:hover': { backgroundColor: '#333' }
                    }}
                  >
                    {loading ? (<CircularProgress size={24} color="inherit" />) : '대본 분석 시작'}
                  </Button>
                </Paper>
              </Grid>

              <Grid item xs={12} md={4}>
                <Card sx={{...cardStyle, mb: 2 }}>
                  <CardContent sx={{ p: 2 }}>
                    <Typography variant="subtitle1" component="h3" fontWeight={600}>Step 1</Typography>
                    <Box sx={{ height: 140, background: '#f0f0f0', mt: 1.5, borderRadius: 2, overflow: 'hidden' }}>
                      <Box
                        component="img"
                        src={scriptResultImg}
                        alt="대본 분석 결과 예시 1"
                        sx={{
                          width: '100%',
                          height: '100%',
                          objectFit: 'cover',
                        }}
                      />
                    </Box>
                    <Typography variant="body2" sx={{ mt: 1.5 }}>
                      주의사항 및 가이드1
                      <br />
                      두 번째 줄 가이드라인입니다.
                    </Typography>
                  </CardContent>
                </Card>
                <Card sx={cardStyle}>
                  <CardContent sx={{ p: 2 }}>
                    <Typography variant="subtitle1" component="h3" fontWeight={600}>Step 2</Typography>
                    <Box sx={{ height: 140, background: '#f0f0f0', mt: 1.5, borderRadius: 2, overflow: 'hidden' }}>
                      <Box
                        component="img"
                        src={scriptResult2Img}
                        alt="대본 분석 결과 예시 2"
                        sx={{
                          width: '100%',
                          height: '100%',
                          objectFit: 'cover',
                        }}
                      />
                    </Box>
                    <Typography variant="body2" sx={{ mt: 1.5 }}>
                      주의사항 및 가이드2
                      <br />
                      두 번째 줄 가이드라인입니다.
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>

            {error && (
              <Alert severity="error" sx={{ mb: 4 }} onClose={() => setError(null)}>
                {error}
              </Alert>
            )}
          </Box>
        </Fade>

        <Box
          component="footer"
          sx={{ mt: 8, textAlign: 'center', borderTop: '1px solid #eee', pt: 4, pb: 2, color: '#666' }}
        >
          <Typography variant="body2">
            © 2025 PRESENT INSIGHT. All rights reserved.
          </Typography>
        </Box>
      </Container>
    </Box>
  );
}