/**
 * UserProfilePage.tsx
 * 사용자 프로필 정보를 표시하고 편집할 수 있는 페이지 컴포넌트
 * 
 * 백엔드 개발자를 위한 가이드:
 * 1. 이 파일은 사용자 프로필 정보를 표시하고 수정하는 페이지를 구현합니다.
 * 2. API 연동 포인트는 주석 "API 호출(백엔드 수정 필요)"로 표시되어 있습니다.
 * 3. 현재는 테스트용 목업 데이터를 사용 중이며, 실제 구현 시 해당 부분을 제거하고 API 응답을 사용해야 합니다.
 * 
 * 기능:
 * - 사용자 프로필 정보 조회 및 표시
 * - 프로필 정보 편집 (이름, 이메일, 비밀번호)
 * - 사용자 활동 통계 및 최근 활동 표시
 */

import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  Container,
  TextField,
  Typography,
  Paper,
  Stack,
  Grid,
  Divider,
  Avatar,
  Fade,
  CircularProgress,
  Alert,
  Card,
  CardContent,
} from '@mui/material';
import { Edit as EditIcon, Save as SaveIcon, ArrowBack as ArrowBackIcon } from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

/**
 * 사용자 정보 타입 정의
 * 
 * 백엔드 개발자 참고사항:
 * - 이 인터페이스는 API 응답의 구조와 일치해야 합니다
 * - name: 사용자 이름 (필수)
 * - email: 이메일 주소 (선택)
 * - joinDate: 가입일자 문자열 (ISO 형식 권장: YYYY-MM-DD)
 * - totalPresentations: 총 발표 분석 횟수
 * - recentActivity: 최근 활동 내역 배열 (선택)
 */
type UserProfile = {
  name: string;
  email?: string | null;
  joinDate: string;
  totalPresentations: number;
  recentActivity?: {
    date: string;      // 활동 날짜 (ISO 형식 권장: YYYY-MM-DD)
    description: string; // 활동 설명
  }[];
};

/**
 * 프로필 편집 폼 데이터 타입 정의
 * 
 * 백엔드 개발자 참고사항:
 * - 이 인터페이스는 프로필 업데이트 API 요청 시 사용되는 데이터 구조입니다
 * - 비밀번호 변경 시 currentPassword, newPassword 필드가 함께 전송됩니다
 */
type ProfileFormData = {
  name: string;
  email: string;
  currentPassword: string;
  newPassword: string;
  confirmPassword: string;
};

/**
 * 사용자 프로필 페이지 컴포넌트
 * 
 * 백엔드 개발자 참고사항:
 * - 페이지 로드 시 사용자 정보를 가져오는 API 호출이 이루어집니다
 * - 프로필 수정 시 업데이트 API 호출이 이루어집니다
 * - 오류 처리 및 성공 메시지 표시 로직이 구현되어 있습니다
 */
const UserProfilePage: React.FC = () => {
  // 기본 상태 관리
  const [profile, setProfile] = useState<UserProfile>({
    name: '',
    email: '',
    joinDate: new Date().toISOString(),
    totalPresentations: 0,
    recentActivity: [],
  });
  
  // UI 상태 관리
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [editMode, setEditMode] = useState(false);
  
  // 폼 데이터 상태
  const [formData, setFormData] = useState<ProfileFormData>({
    name: '',
    email: '',
    currentPassword: '',
    newPassword: '',
    confirmPassword: ''
  });
  
  // 애니메이션 상태
  const [showContent, setShowContent] = useState(false);
  const navigate = useNavigate();
  
  // 페이지 로드 시 애니메이션 적용
  useEffect(() => {
    const timer = setTimeout(() => setShowContent(true), 300);
    return () => clearTimeout(timer);
  }, []);
  
  /**
   * 사용자 정보 불러오기
   * 
   * 백엔드 개발자 참고사항:
   * 1. API 엔드포인트: GET /api/user/profile
   * 2. 인증: 현재 로그인한 사용자의 정보를 반환합니다 (토큰 기반 인증 사용)
   * 3. 응답 형식: UserProfile 타입과 일치해야 합니다
   * 4. 개발 완료 후 아래 목업 데이터 부분을 제거하고 실제 API 응답으로 대체하세요
   */
const fetchUserProfile = async () => {
  setLoading(true);
  const token = localStorage.getItem('token');
  if (!token) {
    setError('로그인이 필요합니다.');
    setLoading(false);
    navigate('/');
    return;
  }
  console.log('Sending token:', token); // 토큰 값 확인

  let combinedProfile: UserProfile = {
    name: '',
    email: '',
    joinDate: new Date().toISOString(),
    totalPresentations: 0,
    recentActivity: [],
  };

  try {
    // Spring Boot 호출
    const springConfig = { headers: { Authorization: `Bearer ${token}` } };
    console.log('Spring Boot request config:', springConfig);
    const userResponse = await axios.get('http://localhost:8080/spring/api/user/profile', springConfig);
    console.log('Spring Boot response:', userResponse.data);

    combinedProfile = {
      ...combinedProfile,
      name: userResponse.data.name || userResponse.data.username || 'Unknown',
      email: userResponse.data.email || '',
      joinDate: userResponse.data.joinDate || new Date().toISOString(),
    };

    // FastAPI 호출 (독립적으로 처리)
    try {
      const fastApiConfig = { headers: { Authorization: `Bearer ${token}` } };
      console.log('FastAPI request config:', fastApiConfig);
      const analysisResponse = await axios.get('http://localhost:5000/fastapi/api/analysis/stats', fastApiConfig);
      console.log('FastAPI response:', analysisResponse.data);
      combinedProfile = {
        ...combinedProfile,
        totalPresentations: analysisResponse.data.totalPresentations || 0,
        recentActivity: analysisResponse.data.recentActivity || [],
      };
    } catch (fastApiErr) {
      console.warn('FastAPI request failed:', fastApiErr);
      // FastAPI 실패 시 기본값 유지
    }

    console.log('Combined profile:', combinedProfile); // 디버깅용

    setProfile(combinedProfile);
    setFormData({
      name: combinedProfile.name,
      email: combinedProfile.email || '',
      currentPassword: '',
      newPassword: '',
      confirmPassword: '',
    });
  } catch (err) {
    console.error('Profile load error:', err);
    if (axios.isAxiosError(err) && err.response) {
      setError(`Profile load failed: ${err.response.status} - ${err.response.data.message || err.message}`);
      console.log('Error response:', err.response.data);
    } else {
      setError('Failed to connect to server.');
    }
    setProfile(combinedProfile);
      setFormData({
        name: combinedProfile.name,
        email: combinedProfile.email || '',
        currentPassword: '',
        newPassword: '',
        confirmPassword: '',
      });
  } finally {
    setLoading(false);
  }
};

  useEffect(() => {
    fetchUserProfile();
  }, [navigate]);
  
//   // 편집 모드 토글
const toggleEditMode = () => {
  if (editMode) {
    setFormData({
      name: profile.name,
      email: profile.email || '',
      currentPassword: '',
      newPassword: '',
      confirmPassword: ''
    });
    setError(null);
  }
  setEditMode(!editMode);
};
  
//   // 폼 입력값 변경 처리
const handleFormChange = (e: React.ChangeEvent<HTMLInputElement>) => {
  const { name, value } = e.target;
  setFormData(prev => ({ ...prev, [name]: value }));
};
  
  /**
   * 프로필 업데이트 처리
   * 
   * 백엔드 개발자 참고사항:
   * 1. API 엔드포인트: PUT /api/user/profile
   * 2. 요청 본문:
   *    { 
   *      name: string,
   *      email: string,
   *      currentPassword?: string,  // 비밀번호 변경 시에만 포함
   *      newPassword?: string       // 비밀번호 변경 시에만 포함
   *    }
   * 3. 응답 코드:
   *    - 200: 성공
   *    - 400: 잘못된 요청 (유효하지 않은 입력)
   *    - 401: 인증 실패 (현재 비밀번호가 올바르지 않음)
   *    - 409: 충돌 (이미 사용 중인 이메일)
   * 4. 비밀번호 변경은 선택사항이며, currentPassword와 newPassword가 모두 제공된 경우에만 처리합니다
   */
  const handleProfileUpdate = async (e: React.FormEvent) => {
    e.preventDefault();
    
     // 클라이언트 측 기본 유효성 검사
    if (!formData.name.trim()) {
      setError('이름은 필수 입력 항목입니다.');
      return;
    }
    
    if (formData.email && !/^\S+@\S+\.\S+$/.test(formData.email)) {
      setError('유효한 이메일 주소를 입력해주세요.');
      return;
    }
    
    // 비밀번호 변경 시 비밀번호 확인 일치 여부만 검사
    if (formData.newPassword && formData.newPassword !== formData.confirmPassword) {
      setError('새 비밀번호와 확인 비밀번호가 일치하지 않습니다.');
      return;
    }
    
    setSubmitting(true);
    setError(null);
    const token = localStorage.getItem('token');
    try {
      const requestData = {
        name: formData.name,
        email: formData.email || null,
        ...(formData.currentPassword && formData.newPassword
          ? { currentPassword: formData.currentPassword, newPassword: formData.newPassword }
          : {}),
      };
      console.log('Sending update request:', requestData);
      const response = await axios.put('http://localhost:8080/spring/api/user/profile', requestData, {
        headers: { Authorization: `Bearer ${token}` },
      });
      console.log('Update response:', response.data);

       // 업데이트 후 즉시 최신 데이터 가져오기
      setProfile(prev => ({
        ...prev,
        name: response.data.name || prev.name,
        email: response.data.email || prev.email,
        joinDate: response.data.joinDate || prev.joinDate,
      }));
      setFormData(prev => ({
        ...prev,
        name: response.data.name || prev.name,
        email: response.data.email || prev.email,
        currentPassword: '',
        newPassword: '',
        confirmPassword: '',
      }));
      
      setSuccess('프로필이 성공적으로 업데이트되었습니다.');
      setEditMode(false);
      
      // 비밀번호 필드 초기화
      setFormData(prev => ({ ...prev, currentPassword: '', newPassword: '', confirmPassword: '' }));
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      if (axios.isAxiosError(err) && err.response) {
        // API 오류 응답에 따른 메시지 처리
        const status = err.response.status;
        if (status === 400) {
          setError('입력한 정보가 유효하지 않습니다.');
        } else if (status === 401) {
          setError('현재 비밀번호가 일치하지 않습니다.');
        } else if (status === 409) {
          setError('이미 사용 중인 이메일입니다.');
        } else {
          setError('프로필 업데이트 중 오류가 발생했습니다.');
        }
      } else {
        setError('서버 연결에 실패했습니다. 잠시 후 다시 시도해주세요.');
      }
      console.error('프로필 업데이트 에러:', err);
    } finally {
      setSubmitting(false);
    }
  };
  
// 이름의 첫 글자 가져오기 (아바타용)
const getInitial = (name: string | null | undefined): string => {
  if (!name || name.trim() === '') {
    return 'U';
  }
  const firstChar = name.trim().charAt(0);
  return firstChar ? firstChar.toUpperCase() : 'U';
};

  return (
    <Box
      component="main"
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
      <Container maxWidth="md">
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

        {/* 뒤로 가기 버튼 */}
        <Box sx={{ mb: 3 }}>
          <Button
            startIcon={<ArrowBackIcon />}
            onClick={() => navigate('/main')}
            sx={{
              color: '#000',
              fontWeight: 500,
              '&:hover': {
                backgroundColor: 'rgba(0, 0, 0, 0.04)'
              }
            }}
            aria-label="메인으로 돌아가기"
          >
            메인으로 돌아가기
          </Button>
        </Box>
        
        <Fade in={showContent} timeout={1000}>
          <Paper
            elevation={0}
            sx={{
              p: { xs: 3, md: 5 },
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
              component="h1"
              gutterBottom
              sx={{
                fontWeight: 700,
                color: '#000',
                mb: 4,
                position: 'relative',
                '&::after': {
                  content: '""',
                  position: 'absolute',
                  bottom: '-8px',
                  left: 0,
                  width: '40px',
                  height: '3px',
                  backgroundColor: '#000'
                }
              }}
            >
              내 정보
            </Typography>
            
            {loading ? (
              <Box sx={{ display: 'flex', justifyContent: 'center', py: 8 }}>
                <CircularProgress size={40} aria-label="프로필 데이터 로딩 중" />
              </Box>
            ) : (
              <React.Fragment>
                {/* 알림 메시지 영역 */}
                {error && (
                  <Alert 
                    severity="error" 
                    sx={{ mb: 3 }}
                    onClose={() => setError(null)}
                  >
                    {error}
                  </Alert>
                )}
                
                {success && (
                  <Alert 
                    severity="success" 
                    sx={{ mb: 3 }}
                    onClose={() => setSuccess(null)}
                  >
                    {success}
                  </Alert>
                )}
                
                <Grid container spacing={4}>
                  {/* 사용자 기본 정보 영역 */}
                  <Grid item xs={12} md={5}>
                    <Box sx={{ textAlign: 'center', mb: { xs: 3, md: 0 } }}>
                      <Avatar
                        sx={{
                          width: 120,
                          height: 120,
                          mx: 'auto',
                          mb: 2,
                          bgcolor: '#000',
                          fontSize: '3rem',
                          fontWeight: 500
                        }}
                        aria-label={`${profile.name || '사용자'}의 아바타`}
                      >
                        {getInitial(profile.name)}
                      </Avatar>
                      
                      <Typography variant="h5" fontWeight={600} gutterBottom>
                        {profile.name || '이름 없음'}
                      </Typography>
                      
                      <Typography variant="body1" color="text.secondary" gutterBottom>
                        {profile.email || '이메일 없음'}
                      </Typography>
                      
                      <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                        가입일: {profile.joinDate ? new Date(profile.joinDate).toLocaleDateString() : '알 수 없음'}
                      </Typography>
                      
                      <Box sx={{ mt: 3 }}>
                        <Button
                          variant={editMode ? "outlined" : "contained"}
                          color={editMode ? "inherit" : "primary"}
                          startIcon={editMode ? <SaveIcon /> : <EditIcon />}
                          onClick={toggleEditMode}
                          sx={{
                            borderRadius: 8,
                            px: 3,
                            py: 1,
                            backgroundColor: editMode ? 'transparent' : '#000',
                            color: editMode ? '#000' : '#fff',
                            border: editMode ? '1px solid #000' : 'none',
                            '&:hover': {
                              backgroundColor: editMode ? 'rgba(0, 0, 0, 0.04)' : '#333'
                            }
                          }}
                          aria-label={editMode ? "편집 모드 취소" : "정보 수정 모드 시작"}
                        >
                          {editMode ? '취소' : '정보 수정'}
                        </Button>
                      </Box>
                    </Box>
                  </Grid>
                  
                  {/* 프로필 정보/수정 영역 */}
                  <Grid item xs={12} md={7}>
                    {editMode ? (
                      <Box component="form" onSubmit={handleProfileUpdate} noValidate>
                        <Stack spacing={3}>
                          <TextField
                            label="이름"
                            name="name"
                            fullWidth
                            required
                            value={formData.name}
                            onChange={handleFormChange}
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
                          
                          <TextField
                            label="이메일"
                            name="email"
                            fullWidth
                            type="email"
                            value={formData.email}
                            onChange={handleFormChange}
                            inputProps={{ 'aria-label': '이메일 입력' }}
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
                          
                          <Divider sx={{ my: 1 }} />
                          
                          <Typography variant="subtitle2" fontWeight={600}>
                            비밀번호 변경 (선택사항)
                          </Typography>
                          
                          <TextField
                            label="현재 비밀번호"
                            name="currentPassword"
                            fullWidth
                            type="password"
                            value={formData.currentPassword}
                            onChange={handleFormChange}
                            inputProps={{ 'aria-label': '현재 비밀번호 입력' }}
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
                          
                          <TextField
                            label="새 비밀번호"
                            name="newPassword"
                            fullWidth
                            type="password"
                            value={formData.newPassword}
                            onChange={handleFormChange}
                            inputProps={{ 'aria-label': '새 비밀번호 입력' }}
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
                          
                          <TextField
                            label="새 비밀번호 확인"
                            name="confirmPassword"
                            fullWidth
                            type="password"
                            value={formData.confirmPassword}
                            onChange={handleFormChange}
                            inputProps={{ 'aria-label': '새 비밀번호 확인 입력' }}
                            error={formData.newPassword !== formData.confirmPassword && formData.confirmPassword !== ''}
                            helperText={formData.newPassword !== formData.confirmPassword && formData.confirmPassword !== '' 
                              ? "비밀번호가 일치하지 않습니다" 
                              : ""
                            }
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
                          
                          <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
                            <Button
                              type="submit"
                              variant="contained"
                              disabled={submitting}
                              startIcon={submitting ? <CircularProgress size={20} /> : <SaveIcon />}
                              sx={{
                                borderRadius: 8,
                                px: 4,
                                py: 1.2,
                                backgroundColor: '#000',
                                color: '#fff',
                                '&:hover': {
                                  backgroundColor: '#333'
                                }
                              }}
                              aria-label="프로필 저장하기"
                            >
                              {submitting ? '저장 중...' : '저장하기'}
                            </Button>
                          </Box>
                        </Stack>
                      </Box>
                    ) : (
                      <Box>
                        {/* 활동 통계 카드
                           * 백엔드 개발자 참고:
                           * - totalPresentations는 사용자가 분석한 총 발표 수를 표시합니다
                           * - recentActivity.length는 최근 활동 수를 표시합니다
                        */}
                        <Card
                          sx={{
                            mb: 3,
                            borderRadius: 3,
                            boxShadow: '0 4px 12px rgba(0,0,0,0.05)',
                            border: '1px solid #f0f0f0'
                          }}
                        >
                          <CardContent sx={{ p: 3 }}>
                            <Typography variant="h6" fontWeight={600} gutterBottom>
                              활동 통계
                            </Typography>
                            
                            <Grid container spacing={2} sx={{ mt: 1 }}>
                              <Grid item xs={6}>
                                <Box sx={{ textAlign: 'center', p: 2 }}>
                                  <Typography variant="h4" fontWeight={700}>
                                    {profile.totalPresentations}
                                  </Typography>
                                  <Typography variant="body2" color="text.secondary">
                                    총 발표 분석
                                  </Typography>
                                </Box>
                              </Grid>
                              <Grid item xs={6}>
                                <Box sx={{ textAlign: 'center', p: 2 }}>
                                  <Typography variant="h4" fontWeight={700}>
                                    {profile.recentActivity ? profile.recentActivity.length : 0}
                                  </Typography>
                                  <Typography variant="body2" color="text.secondary">
                                    최근 활동
                                  </Typography>
                                </Box>
                              </Grid>
                            </Grid>
                          </CardContent>
                        </Card>
                        
                        {/* 최근 활동 카드
                           * 백엔드 개발자 참고:
                           * - recentActivity 배열의 각 항목은 date와 description을 포함해야 합니다
                           * - 각 활동을 클릭하면 해당 분석 페이지로 이동합니다 (index 기반)
                           * - recentActivity는 날짜 기준 내림차순으로 정렬되어야 합니다
                        */}
                        {profile.recentActivity && profile.recentActivity.length > 0 && (
                          <Card
                            sx={{
                              borderRadius: 3,
                              boxShadow: '0 4px 12px rgba(0,0,0,0.05)',
                              border: '1px solid #f0f0f0'
                            }}
                          >
                            <CardContent sx={{ p: 3 }}>
                              <Typography variant="h6" fontWeight={600} gutterBottom>
                                최근 활동
                              </Typography>
                              
                              <Stack spacing={2} sx={{ mt: 2 }}>
                                {profile.recentActivity.map((activity, index) => (
                                  <Box 
                                    key={index} 
                                    sx={{ 
                                      p: 2, 
                                      borderRadius: 2, 
                                      bgcolor: '#f5f5f5',
                                      transition: 'background-color 0.2s ease',
                                      '&:hover': {
                                        bgcolor: '#f0f0f0',
                                        cursor: 'pointer'
                                      }
                                    }}
                                    onClick={() => navigate(`/analysis/${index}`)} // 사용자가 활동을 클릭하면 해당 분석 페이지로 이동
                                  >
                                    <Typography variant="body2" fontWeight={500}>
                                      {activity.description}
                                    </Typography>
                                    <Typography variant="caption" color="text.secondary">
                                      {new Date(activity.date).toLocaleDateString()}
                                    </Typography>
                                  </Box>
                                ))}
                              </Stack>
                            </CardContent>
                          </Card>
                        )}
                      </Box>
                    )}
                  </Grid>
                </Grid>
              </React.Fragment>
            )}
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

export default UserProfilePage;