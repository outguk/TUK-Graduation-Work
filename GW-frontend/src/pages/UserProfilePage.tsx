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
  Chip,
} from '@mui/material';
import { 
  Edit as EditIcon, 
  Save as SaveIcon, 
  ArrowBack as ArrowBackIcon,
  Movie as MovieIcon,
  TextSnippet as TextSnippetIcon
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

/**
 * 사용자 정보 타입 정의 - 수정된 버전
 */
type UserProfile = {
  name: string;
  email?: string | null;
  joinDate: string;
  totalVideoPresentations: number;
  totalScriptAnalyses: number;
  recentActivity?: {
    date: string;
    description: string;
    type: 'video' | 'script';
    id: string; // 분석 결과 ID
  }[];
};

/**
 * 프로필 편집 폼 데이터 타입 정의
 */
type ProfileFormData = {
  name: string;
  email: string;
  currentPassword: string;
  newPassword: string;
  confirmPassword: string;
};

const UserProfilePage: React.FC = () => {
  // 기본 상태 관리
  const [profile, setProfile] = useState<UserProfile>({
    name: '',
    email: '',
    joinDate: new Date().toISOString(),
    totalVideoPresentations: 0,
    totalScriptAnalyses: 0,
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
   * 사용자 정보 불러오기 - 수정된 버전
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

    let combinedProfile: UserProfile = {
      name: '',
      email: '',
      joinDate: new Date().toISOString(),
      totalVideoPresentations: 0,
      totalScriptAnalyses: 0,
      recentActivity: [],
    };

    try {
      // 1. Spring Boot에서 사용자 기본 정보 가져오기
      const springConfig = { headers: { Authorization: `Bearer ${token}` } };
      const userResponse = await axios.get('/spring/api/user/profile', springConfig);

      combinedProfile = {
        ...combinedProfile,
        name: userResponse.data.name || userResponse.data.username || 'Unknown',
        email: userResponse.data.email || '',
        joinDate: userResponse.data.joinDate || new Date().toISOString(),
      };

      // 2. FastAPI에서 분석 통계 가져오기
      try {
        const fastApiConfig = { headers: { Authorization: `Bearer ${token}` } };
        const analysisResponse = await axios.get('/spring/api/analysis/stats', fastApiConfig);
        
        console.log('Analysis stats response:', analysisResponse.data);

        // FastAPI 응답 구조에 맞게 데이터 매핑
        const statsData = analysisResponse.data;
        
        combinedProfile = {
          ...combinedProfile,
          totalVideoPresentations: statsData.totalVideoPresentations || 0,
          totalScriptAnalyses: statsData.totalScriptAnalyses || 0,
        };

        // 3. 최근 활동 데이터 통합 처리
        const recentVideoActivity = (statsData.recentVideoActivity || []).map((item: any) => ({
          date: item.date || new Date().toISOString(),
          description: `영상 분석: ${item.description || 'Unknown'}`,
          type: 'video' as const,
          id: item.description || 'unknown', // filename 사용
        }));

        const recentScriptActivity = (statsData.recentScriptActivity || []).map((item: any) => ({
          date: item.date || new Date().toISOString(),
          description: `대본 분석: ${item.description || 'Unknown'}`,
          type: 'script' as const,
          id: item.description || 'unknown', // script_id 또는 filename 사용
        }));

        // 모든 활동을 날짜순으로 정렬하여 통합
        const allActivities = [...recentVideoActivity, ...recentScriptActivity]
          .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
          .slice(0, 10); // 최근 10개만 표시

        combinedProfile.recentActivity = allActivities;

      } catch (fastApiErr) {
        console.warn('FastAPI 통계 요청 실패:', fastApiErr);
        
        // FastAPI 실패 시 대안: Spring Boot를 통해 분석 목록 직접 가져오기
        try {
          const analysesResponse = await axios.get('/spring/api/my-analyses', springConfig);
          console.log('My analyses response:', analysesResponse.data);
          
          const videoAnalyses = analysesResponse.data.video_analyses || [];
          const scriptAnalyses = analysesResponse.data.script_analyses || [];
          
          combinedProfile.totalVideoPresentations = videoAnalyses.length;
          combinedProfile.totalScriptAnalyses = scriptAnalyses.length;
          
          // 최근 활동 생성
          const recentVideoActivity = videoAnalyses
            .slice(0, 5)
            .map((item: any) => ({
              date: item.timestamp || new Date().toISOString(),
              description: `영상 분석: ${item.filename?.split('.')[0] || 'Unknown'}`,
              type: 'video' as const,
              id: item.filename || item._id,
            }));

          const recentScriptActivity = scriptAnalyses
            .slice(0, 5)
            .map((item: any) => ({
              date: item.timestamp || new Date().toISOString(),
              description: `대본 분석: ${item.filename?.split('.')[0] || 'Unknown'}`,
              type: 'script' as const,
              id: item._id,
            }));

          const allActivities = [...recentVideoActivity, ...recentScriptActivity]
            .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
            .slice(0, 10);

          combinedProfile.recentActivity = allActivities;
          
        } catch (myAnalysesErr) {
          console.warn('My analyses 요청도 실패:', myAnalysesErr);
        }
      }

      console.log('최종 프로필 데이터:', combinedProfile);

      setProfile(combinedProfile);
      setFormData({
        name: combinedProfile.name,
        email: combinedProfile.email || '',
        currentPassword: '',
        newPassword: '',
        confirmPassword: '',
      });
      setError(null);

    } catch (err) {
      console.error('프로필 로드 에러:', err);
      if (axios.isAxiosError(err) && err.response) {
        setError(`프로필 로드 실패: ${err.response.status} - ${err.response.data.message || err.message}`);
      } else {
        setError('서버 연결에 실패했습니다.');
      }
      
      // 오류 발생 시에도 기본 프로필 설정
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
  
  // 편집 모드 토글
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
  
  // 폼 입력값 변경 처리
  const handleFormChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };
  
  /**
   * 프로필 업데이트 처리
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
      
      const response = await axios.put('/spring/api/user/profile', requestData, {
        headers: { Authorization: `Bearer ${token}` },
      });

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
      
      setTimeout(() => setSuccess(null), 3000);
    } catch (err) {
      if (axios.isAxiosError(err) && err.response) {
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

  // 활동 클릭 시 해당 분석 페이지로 이동
  const handleActivityClick = (activity: {
    date: string;
    description: string;
    type: 'video' | 'script';
    id: string;
  }) => {
    if (activity.type === 'video') {
      // 영상 분석 결과 페이지로 이동
      navigate(`/analysis/${activity.id}`);
    } else if (activity.type === 'script') {
      // 대본 분석 결과 페이지로 이동
      navigate(`/analysis/${activity.id}`);
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
                        {/* 활동 통계 카드 - 수정된 버전 */}
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
                              <Grid item xs={4}>
                                <Box sx={{ textAlign: 'center', p: 2 }}>
                                  <Typography variant="h4" fontWeight={700}>
                                    {profile.totalVideoPresentations}
                                  </Typography>
                                  <Typography variant="body2" color="text.secondary">
                                    영상 분석
                                  </Typography>
                                </Box>
                              </Grid>
                              <Grid item xs={4}>
                                <Box sx={{ textAlign: 'center', p: 2 }}>
                                  <Typography variant="h4" fontWeight={700}>
                                    {profile.totalScriptAnalyses}
                                  </Typography>
                                  <Typography variant="body2" color="text.secondary">
                                    대본 분석
                                  </Typography>
                                </Box>
                              </Grid>
                              <Grid item xs={4}>
                                <Box sx={{ textAlign: 'center', p: 2 }}>
                                  <Typography variant="h4" fontWeight={700}>
                                    {profile.totalVideoPresentations + profile.totalScriptAnalyses}
                                  </Typography>
                                  <Typography variant="body2" color="text.secondary">
                                    총 분석
                                  </Typography>
                                </Box>
                              </Grid>
                            </Grid>
                          </CardContent>
                        </Card>
                        
                        {/* 최근 활동 카드 - 수정된 버전 */}
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
                                최근 활동 ({profile.recentActivity.length}개)
                              </Typography>
                              
                              <Stack spacing={2} sx={{ mt: 2 }}>
                                {profile.recentActivity.map((activity, index) => (
                                  <Box 
                                    key={index} 
                                    sx={{ 
                                      p: 2, 
                                      borderRadius: 2, 
                                      bgcolor: '#f5f5f5',
                                      transition: 'all 0.2s ease',
                                      cursor: 'pointer',
                                      display: 'flex',
                                      alignItems: 'center',
                                      gap: 2,
                                      '&:hover': {
                                        bgcolor: '#f0f0f0',
                                        transform: 'translateY(-1px)',
                                        boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
                                      }
                                    }}
                                    onClick={() => handleActivityClick(activity)}
                                  >
                                    {/* 활동 타입 아이콘 */}
                                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                                      {activity.type === 'video' ? (
                                        <MovieIcon sx={{ color: '#1976d2', fontSize: 24 }} />
                                      ) : (
                                        <TextSnippetIcon sx={{ color: '#ed6c02', fontSize: 24 }} />
                                      )}
                                    </Box>
                                    
                                    {/* 활동 내용 */}
                                    <Box sx={{ flexGrow: 1 }}>
                                      <Typography variant="body1" fontWeight={500}>
                                        {activity.description}
                                      </Typography>
                                      <Typography variant="caption" color="text.secondary">
                                        {new Date(activity.date).toLocaleDateString('ko-KR', {
                                          year: 'numeric',
                                          month: 'long',
                                          day: 'numeric',
                                          hour: '2-digit',
                                          minute: '2-digit'
                                        })}
                                      </Typography>
                                    </Box>
                                    
                                    {/* 활동 타입 칩 */}
                                    <Chip 
                                      label={activity.type === 'video' ? '영상' : '대본'}
                                      size="small"
                                      color={activity.type === 'video' ? 'primary' : 'warning'}
                                      variant="outlined"
                                    />
                                  </Box>
                                ))}
                              </Stack>
                              
                              {/* 더 많은 활동 보기 버튼 */}
                              <Box sx={{ mt: 3, textAlign: 'center' }}>
                                <Button
                                  variant="outlined"
                                  onClick={() => navigate('/analysis')}
                                  sx={{
                                    borderColor: '#000',
                                    color: '#000',
                                    '&:hover': {
                                      borderColor: '#333',
                                      backgroundColor: 'rgba(0, 0, 0, 0.04)'
                                    }
                                  }}
                                >
                                  모든 분석 결과 보기
                                </Button>
                              </Box>
                            </CardContent>
                          </Card>
                        )}
                        
                        {/* 활동이 없는 경우 */}
                        {(!profile.recentActivity || profile.recentActivity.length === 0) && (
                          <Card
                            sx={{
                              borderRadius: 3,
                              boxShadow: '0 4px 12px rgba(0,0,0,0.05)',
                              border: '1px solid #f0f0f0'
                            }}
                          >
                            <CardContent sx={{ p: 3, textAlign: 'center' }}>
                              <Typography variant="h6" fontWeight={600} gutterBottom>
                                아직 활동이 없습니다
                              </Typography>
                              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                                첫 번째 발표를 업로드하거나 대본을 분석해보세요!
                              </Typography>
                              <Stack direction="row" spacing={2} justifyContent="center">
                                <Button
                                  variant="contained"
                                  onClick={() => navigate('/upload')}
                                  sx={{
                                    backgroundColor: '#000',
                                    '&:hover': { backgroundColor: '#333' }
                                  }}
                                >
                                  영상 업로드
                                </Button>
                                <Button
                                  variant="outlined"
                                  onClick={() => navigate('/analysis/script-upload')}
                                  sx={{
                                    borderColor: '#000',
                                    color: '#000',
                                    '&:hover': {
                                      borderColor: '#333',
                                      backgroundColor: 'rgba(0, 0, 0, 0.04)'
                                    }
                                  }}
                                >
                                  대본 분석
                                </Button>
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