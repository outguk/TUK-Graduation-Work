// import React, { useState, useEffect } from 'react';
// import {
//   Box,
//   Button,
//   Container,
//   TextField,
//   Typography,
//   Link as MuiLink,
//   Stack,
//   Paper,
//   Fade,
// } from '@mui/material';
// import { useNavigate } from 'react-router-dom';
// import axios from 'axios';

// const SignInPage: React.FC = () => {
//   const [username, setUsername] = useState('');
//   const [password, setPassword] = useState('');
//   const [error, setError] = useState<string | null>(null);
//   const navigate = useNavigate();
  
//   // 애니메이션 상태 관리
//   const [showContent, setShowContent] = useState(false);
  
//   useEffect(() => {
//     // 애니메이션 시작
//     const timer = setTimeout(() => setShowContent(true), 300);
//     return () => clearTimeout(timer);
//   }, []);

//   const handleLogin = async (e: React.FormEvent) => {
//     e.preventDefault();
//     try {
//       const response = await axios.post('http://localhost:8080/spring/api/users/login', { username, password });
//       if (response.status === 200) {
//         const token = response.data.token; // 응답에서 토큰 추출
//         localStorage.setItem('token', token); // 토큰 저장
//         console.log('Login successful, token saved:', token); // 디버깅 로그 추가
//         console.log('Stored token in localStorage:', localStorage.getItem('token')); // 저장 확인
//         navigate('/main');
//       }
//     } catch (err) {
//       if (axios.isAxiosError(err) && err.response) {
//         setError(err.response.data.message || '로그인 실패! username 또는 비밀번호를 확인해주세요.');
//         console.error('Login error response:', err.response.data); // 에러 상세 출력
//       } else {
//         setError('서버 연결에 실패했습니다.');
//         console.error('Login failed, no response:', err); // 네트워크 에러 출력
//       }
//     }
//   };

//   return (
//     <Box
//       sx={{
//         minHeight: '100vh',
//         display: 'flex',
//         flexDirection: 'column',
//         alignItems: 'center',
//         justifyContent: 'center',
//         py: { xs: 6, md: 8 },
//         background: '#FFFFFF',
//       }}
//     >
//       <Container maxWidth="sm">
//         {/* 로고 및 브랜드 요소 */}
//         <Box
//           sx={{
//             display: 'flex',
//             justifyContent: 'center',
//             mb: 5
//           }}
//         >
//           <Typography 
//             variant="h6" 
//             fontWeight={700}
//             sx={{
//               letterSpacing: 1.2,
//               color: '#000',
//               p: 1,
//               borderBottom: '2px solid #000'
//             }}
//           >
//             PRESENT INSIGHT
//           </Typography>
//         </Box>
        
//         <Fade in={showContent} timeout={1000}>
//           <Paper
//             elevation={0}
//             sx={{
//             p: 5,
//             borderRadius: 4,
//             boxShadow: '0 10px 30px rgba(0,0,0,0.08)',
//             border: '1px solid #eaeaea',
//             position: 'relative',
//             overflow: 'hidden',
//             '&::before': {
//               content: '""',
//               position: 'absolute',
//               top: 0,
//               left: 0,
//               width: '100%',
//               height: '4px',
//               background: '#000',
//             }
//           }}
//         >
//           <Typography 
//             variant="h4" 
//             align="center" 
//             gutterBottom
//             sx={{
//               fontWeight: 700,
//               color: '#000',
//               mb: 3,
//               position: 'relative'
//             }}
//           >
//             로그인
//           </Typography>
          
//           <Typography 
//             variant="subtitle1" 
//             align="center" 
//             color="text.secondary"
//             sx={{ mb: 4 }}
//           >
//             발표 분석을 시작하려면 로그인하세요
//           </Typography>
          
//           <Box component="form" onSubmit={handleLogin} noValidate role="form">
//             <Stack spacing={3}>
//               <TextField
//                 label="이름"
//                 fullWidth
//                 required
//                 value={username}
//                 onChange={(e) => setUsername(e.target.value)}
//                 inputProps={{ 'aria-label': '이름 입력' }}
//                 variant="outlined"
//                 sx={{
//                   '& .MuiOutlinedInput-root': {
//                     borderRadius: 2,
//                     '&:hover fieldset': {
//                       borderColor: '#000',
//                     },
//                     '&.Mui-focused fieldset': {
//                       borderColor: '#000',
//                     }
//                   },
//                   '& .MuiFormLabel-root.Mui-focused': {
//                     color: '#000',
//                   }
//                 }}
//               />
//               <TextField
//                 label="비밀번호"
//                 fullWidth
//                 required
//                 type="password"
//                 value={password}
//                 onChange={(e) => setPassword(e.target.value)}
//                 inputProps={{ 'aria-label': '비밀번호 입력' }}
//                 variant="outlined"
//                 sx={{
//                   '& .MuiOutlinedInput-root': {
//                     borderRadius: 2,
//                     '&:hover fieldset': {
//                       borderColor: '#000',
//                     },
//                     '&.Mui-focused fieldset': {
//                       borderColor: '#000',
//                     }
//                   },
//                   '& .MuiFormLabel-root.Mui-focused': {
//                     color: '#000',
//                   }
//                 }}
//               />
//               <Button 
//                 type="submit" 
//                 fullWidth 
//                 variant="contained" 
//                 size="large" 
//                 aria-label="로그인"
//                 sx={{ 
//                   mt: 2,
//                   py: 1.5, 
//                   borderRadius: 10, 
//                   fontWeight: 600, 
//                   fontSize: '1rem',
//                   backgroundColor: '#000',
//                   color: '#fff',
//                   boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
//                   transition: 'all 0.3s ease',
//                   '&:hover': {
//                     boxShadow: '0 6px 15px rgba(0, 0, 0, 0.25)',
//                     transform: 'translateY(-3px)',
//                     backgroundColor: '#333'
//                   }
//                 }}
//               >
//                 로그인
//               </Button>
//             </Stack>
//           </Box>
          
//           {error && (
//             <Typography 
//               color="error" 
//               mt={3} 
//               role="alert"
//               sx={{ 
//                 textAlign: 'center',
//                 fontSize: '0.875rem',
//                 bgcolor: 'rgba(244, 67, 54, 0.08)',
//                 p: 1.5,
//                 borderRadius: 1
//               }}
//             >
//               {error}
//             </Typography>
//           )}
          
//           <Box 
//             textAlign="center" 
//             mt={4}
//             sx={{
//               display: 'flex',
//               justifyContent: 'center',
//               gap: 3
//             }}
//           >
//             <MuiLink 
//               href="/signup" 
//               underline="none" 
//               aria-label="회원 가입 페이지로 이동"
//               sx={{ 
//                 color: '#000',
//                 fontWeight: 500,
//                 position: 'relative',
//                 '&::after': {
//                   content: '""',
//                   position: 'absolute',
//                   width: '0%',
//                   height: '2px',
//                   bottom: '-4px',
//                   left: 0,
//                   backgroundColor: '#000',
//                   transition: 'width 0.3s ease'
//                 },
//                 '&:hover::after': {
//                   width: '100%'
//                 }
//               }}
//             >
//               회원 가입
//             </MuiLink>
            
//             <MuiLink 
//               href="/users" 
//               underline="none" 
//               aria-label="회원 목록 페이지로 이동"
//               sx={{ 
//                 color: '#000',
//                 fontWeight: 500,
//                 position: 'relative',
//                 '&::after': {
//                   content: '""',
//                   position: 'absolute',
//                   width: '0%',
//                   height: '2px',
//                   bottom: '-4px',
//                   left: 0,
//                   backgroundColor: '#000',
//                   transition: 'width 0.3s ease'
//                 },
//                 '&:hover::after': {
//                   width: '100%'
//                 }
//               }}
//             >
//               회원 목록
//             </MuiLink>
//           </Box>
//         </Paper>
//         </Fade>
        
//         {/* 푸터 */}
//         <Box 
//           component="footer"
//           sx={{ 
//             mt: 8, 
//             textAlign: 'center', 
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

// export default SignInPage;

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

const SignInPage: React.FC = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();
  
  const [showContent, setShowContent] = useState(false);
  
  useEffect(() => {
    const timer = setTimeout(() => setShowContent(true), 300);
    return () => clearTimeout(timer);
  }, []);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const response = await axios.post('http://localhost:8080/spring/api/users/login', { username, password });
      if (response.status === 200) {
        const token = response.data.token;
        if (!token) {
          throw new Error('서버에서 토큰이 반환되지 않았습니다.');
        }
        localStorage.setItem('token', token);
        console.log('Login successful, token saved:', token);
        console.log('Stored token in localStorage:', localStorage.getItem('token'));
        
        const storedToken = localStorage.getItem('token');
        if (storedToken !== token) {
          throw new Error('토큰이 localStorage에 저장되지 않았습니다.');
        }
        
        navigate('/main');
      }
    } catch (err: unknown) {  // err를 unknown으로 타입 선언
      if (axios.isAxiosError(err) && err.response) {
        setError(err.response.data.message || '로그인 실패! username 또는 비밀번호를 확인해주세요.');
        console.error('Login error response:', err.response.data);
      } else {
        // err가 Error 객체인지 확인 후 안전하게 message 접근
        const errorMessage = err instanceof Error ? err.message : '서버 연결에 실패했습니다.';
        setError(errorMessage);
        console.error('Login failed:', err);
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
        <Box sx={{ display: 'flex', justifyContent: 'center', mb: 5 }}>
          <Typography 
            variant="h6" 
            fontWeight={700}
            sx={{ letterSpacing: 1.2, color: '#000', p: 1, borderBottom: '2px solid #000' }}
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
            <Typography variant="h4" align="center" gutterBottom sx={{ fontWeight: 700, color: '#000', mb: 3 }}>
              로그인
            </Typography>
            <Typography variant="subtitle1" align="center" color="text.secondary" sx={{ mb: 4 }}>
              발표 분석을 시작하려면 로그인하세요
            </Typography>
            
            <Box component="form" onSubmit={handleLogin} noValidate role="form">
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
                      '&:hover fieldset': { borderColor: '#000' },
                      '&.Mui-focused fieldset': { borderColor: '#000' },
                    },
                    '& .MuiFormLabel-root.Mui-focused': { color: '#000' },
                  }}
                />
                <Button 
                  type="submit" 
                  fullWidth 
                  variant="contained" 
                  size="large" 
                  aria-label="로그인"
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
                    '&:hover': { boxShadow: '0 6px 15px rgba(0, 0, 0, 0.25)', transform: 'translateY(-3px)', backgroundColor: '#333' }
                  }}
                >
                  로그인
                </Button>
              </Stack>
            </Box>
            
            {error && (
              <Typography 
                color="error" 
                mt={3} 
                role="alert"
                sx={{ textAlign: 'center', fontSize: '0.875rem', bgcolor: 'rgba(244, 67, 54, 0.08)', p: 1.5, borderRadius: 1 }}
              >
                {error}
              </Typography>
            )}
            
            <Box textAlign="center" mt={4} sx={{ display: 'flex', justifyContent: 'center', gap: 3 }}>
              <MuiLink href="/signup" underline="none" aria-label="회원 가입 페이지로 이동" sx={{ color: '#000', fontWeight: 500, position: 'relative', '&::after': { content: '""', position: 'absolute', width: '0%', height: '2px', bottom: '-4px', left: 0, backgroundColor: '#000', transition: 'width 0.3s ease' }, '&:hover::after': { width: '100%' } }}>
                회원 가입
              </MuiLink>
              <MuiLink href="/users" underline="none" aria-label="회원 목록 페이지로 이동" sx={{ color: '#000', fontWeight: 500, position: 'relative', '&::after': { content: '""', position: 'absolute', width: '0%', height: '2px', bottom: '-4px', left: 0, backgroundColor: '#000', transition: 'width 0.3s ease' }, '&:hover::after': { width: '100%' } }}>
                회원 목록
              </MuiLink>
            </Box>
          </Paper>
        </Fade>
        
        <Box component="footer" sx={{ mt: 8, textAlign: 'center', color: '#666' }}>
          <Typography variant="body2">
            © 2025 PRESENT INSIGHT. All rights reserved.
          </Typography>
        </Box>
      </Container>
    </Box>
  );
};

export default SignInPage;