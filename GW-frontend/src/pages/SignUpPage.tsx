import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Container,
  TextField,
  Typography,
  Link as MuiLink,
  Stack,
  Paper,
  Fade,
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const SignUpPage: React.FC = () => {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');   // 이메일  추가
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [createdAt, setCreatedAt] = useState<string | null>(null);  // 가입일자 상태 추가
  const navigate = useNavigate();
  
  // 애니메이션 상태 관리
  const [showContent, setShowContent] = useState(false);
  
  useEffect(() => {
    // 애니메이션 시작
    const timer = setTimeout(() => setShowContent(true), 300);
    return () => clearTimeout(timer);
  }, []);

  const handleSignUp = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const response = await axios.post('/spring/api/users/new', { username, email, password });
      if (response.status === 200 || response.status === 201) { // 201 Created도 허용
        const token = response.data.token; // 응답에서 토큰 추출
        localStorage.setItem('token', token); // 토큰 저장
        setCreatedAt(response.data.createdAt || new Date().toISOString());
        navigate('/');
      }
    } catch (err) {
      if (axios.isAxiosError(err) && err.response) {
        setError(err.response.data.message || '회원가입 중 오류가 발생했습니다.');
      } else {
        setError('서버 연결에 실패했습니다.');
      }
    }
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        py: { xs: 6, md: 8 },
        background: '#FFFFFF',
      }}
    >
      <Container maxWidth="sm">
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
          <Paper
            elevation={0}
            sx={{
            p: 5,
            borderRadius: 4,
            boxShadow: '0 10px 30px rgba(0,0,0,0.08)',
            border: '1px solid #eaeaea',
            position: 'relative',
            overflow: 'hidden',
            '&::before': {
              content: '""',
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '4px',
              background: '#000',
            }
          }}
        >
          <Typography 
            variant="h4" 
            align="center" 
            gutterBottom
            sx={{
              fontWeight: 700,
              color: '#000',
              mb: 3,
              position: 'relative',
              '&::after': {
                content: '""',
                position: 'absolute',
                bottom: '-8px',
                left: '50%',
                transform: 'translateX(-50%)',
                width: '40px',
                height: '3px',
                backgroundColor: '#000'
              }
            }}
          >
            회원 가입
          </Typography>
          
          <Typography 
            variant="subtitle1" 
            align="center" 
            color="text.secondary"
            sx={{ mb: 4 }}
          >
            내 발표 분석을 위한 계정을 생성하세요
          </Typography>
          
          <Box component="form" onSubmit={handleSignUp} noValidate role="form">
            <Stack spacing={3}>
              <TextField
                label="이름"
                fullWidth
                required
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                inputProps={{ 'aria-label': '이름 입력' }}
                variant="outlined"
                sx={{
                  '& .MuiOutlinedInput-root': {
                    borderRadius: 2,
                    '&:hover fieldset': {
                      borderColor: '#000',
                    },
                    '&.Mui-focused fieldset': {
                      borderColor: '#000',
                    }
                  },
                  '& .MuiFormLabel-root.Mui-focused': {
                    color: '#000',
                  }
                }}
              />
              {/* 이메일 */}
              <TextField
                  label="이메일"
                  fullWidth
                  required
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  inputProps={{ 'aria-label': '이메일 입력' }}
                  variant="outlined"
                  sx={{
                    '& .MuiOutlinedInput-root': {
                      borderRadius: 2,
                      '&:hover fieldset': { borderColor: '#000' },
                      '&.Mui-focused fieldset': { borderColor: '#000' },
                    },
                    '& .MuiFormLabel-root.Mui-focused': { color: '#000' },
                  }}
                />
              <TextField
                label="비밀번호"
                fullWidth
                required
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                inputProps={{ 'aria-label': '비밀번호 입력' }}
                variant="outlined"
                sx={{
                  '& .MuiOutlinedInput-root': {
                    borderRadius: 2,
                    '&:hover fieldset': {
                      borderColor: '#000',
                    },
                    '&.Mui-focused fieldset': {
                      borderColor: '#000',
                    }
                  },
                  '& .MuiFormLabel-root.Mui-focused': {
                    color: '#000',
                  }
                }}
              />
              <Button 
                type="submit" 
                fullWidth 
                variant="contained" 
                size="large" 
                aria-label="회원가입 등록"
                sx={{ 
                  mt: 2,
                  py: 1.5, 
                  borderRadius: 10, 
                  fontWeight: 600, 
                  fontSize: '1rem',
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
                등록
              </Button>
            </Stack>
          </Box>
          
          {error && (
            <Typography 
              color="error" 
              mt={3} 
              role="alert"
              sx={{ 
                textAlign: 'center',
                fontSize: '0.875rem',
                bgcolor: 'rgba(244, 67, 54, 0.08)',
                p: 1.5,
                borderRadius: 1
              }}
            >
              {error}
            </Typography>
          )}

          {/* 가입일자 표시 */}
          {createdAt && (
              <Typography 
                color="text.secondary" 
                mt={3} 
                sx={{ textAlign: 'center', fontSize: '0.875rem' }}
              >
                가입일자: {new Date(createdAt).toLocaleString()}
              </Typography>
          )}
          
          <Box 
            textAlign="center" 
            mt={4}
          >
            <MuiLink 
              href="/" 
              underline="none" 
              aria-label="홈으로 돌아가기"
              sx={{ 
                color: '#000',
                fontWeight: 500,
                position: 'relative',
                '&::after': {
                  content: '""',
                  position: 'absolute',
                  width: '0%',
                  height: '2px',
                  bottom: '-4px',
                  left: 0,
                  backgroundColor: '#000',
                  transition: 'width 0.3s ease'
                },
                '&:hover::after': {
                  width: '100%'
                }
              }}
            >
              홈으로 돌아가기
            </MuiLink>
          </Box>
        </Paper>
        </Fade>
        
        {/* 푸터 */}
        <Box 
          component="footer"
          sx={{ 
            mt: 8, 
            textAlign: 'center', 
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

export default SignUpPage;