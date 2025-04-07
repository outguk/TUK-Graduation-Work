import React, { useState } from 'react';
import {
  Box,
  Button,
  Container,
  TextField,
  Typography,
  Link as MuiLink,
  Stack,
} from '@mui/material';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const SignInPage: React.FC = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      // 스프링에 /users/login POST → { username, password } 전송
      const response = await axios.post('/users/login', { username, password });
      if (response.status === 200) {
        navigate('/upload');
      }
    } catch (err) {
      setError('로그인 실패! username 또는 비밀번호를 확인해주세요.');
    }
  };

  return (
    <Container maxWidth="sm">
      <Box
        component="main"
        sx={{
          mt: 8,
          p: 4,
          bgcolor: 'background.paper',
          borderRadius: 2,
          boxShadow: 3,
        }}
      >
        <Typography variant="h4" gutterBottom align="center" color="primary">
          Hello TUK
        </Typography>
        <Typography variant="subtitle1" align="center" color="text.secondary">
          회원 기능
        </Typography>
        <Box component="form" onSubmit={handleLogin} noValidate role="form" sx={{ mt: 3 }}>
          <Stack spacing={2}>
            <TextField
              label="아이디"
              fullWidth
              required
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              inputProps={{ 'aria-label': '아이디 입력' }}
            />
            <TextField
              label="비밀번호"
              fullWidth
              required
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              inputProps={{ 'aria-label': '비밀번호 입력' }}
            />
            <Button type="submit" fullWidth variant="contained" size="large" aria-label="로그인">
              로그인
            </Button>
          </Stack>
        </Box>
        {error && (
          <Typography color="error" mt={2} role="alert">
            {error}
          </Typography>
        )}
        <Box textAlign="center" mt={3}>
          <MuiLink href="/signup" underline="hover" sx={{ mx: 1 }} aria-label="회원 가입 페이지로 이동">
            회원 가입
          </MuiLink>
          <MuiLink href="/users" underline="hover" sx={{ mx: 1 }} aria-label="회원 목록 페이지로 이동">
            회원 목록
          </MuiLink>
        </Box>
      </Box>
    </Container>
  );
};

export default SignInPage;
