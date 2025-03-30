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

const SignUpPage: React.FC = () => {
  const [name, setName] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  const handleSignUp = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      // JSON 형태로 데이터 보냄
      const response = await axios.post('/users/new', { name, password });
      if (response.status === 200 || response.status === 302) {
        navigate('/');
      }
    } catch (err) {
      setError('회원가입 중 오류가 발생했습니다.');
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
          회원 가입
        </Typography>
        <Box component="form" onSubmit={handleSignUp} noValidate role="form" sx={{ mt: 3 }}>
          <Stack spacing={2}>
            <TextField
              label="이름"
              variant="outlined"
              fullWidth
              required
              value={name}
              onChange={(e) => setName(e.target.value)}
              inputProps={{ 'aria-label': '이름 입력' }}
            />
            <TextField
              label="비밀번호"
              variant="outlined"
              fullWidth
              required
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              inputProps={{ 'aria-label': '비밀번호 입력' }}
            />
            <Button type="submit" fullWidth variant="contained" size="large" aria-label="회원가입 등록">
              등록
            </Button>
          </Stack>
        </Box>
        {error && (
          <Typography color="error" mt={2} role="alert">
            {error}
          </Typography>
        )}
        <Box textAlign="center" mt={3}>
          <MuiLink href="/" underline="hover" aria-label="홈으로 돌아가기">
            돌아가기
          </MuiLink>
        </Box>
      </Box>
    </Container>
  );
};

export default SignUpPage;
