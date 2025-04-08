import { useState, useEffect } from 'react';
import Grid from '@mui/material/Grid';
import { Box, Typography, Button, Card, CardContent, Fade, Grow, Container } from "@mui/material";
import { PlayCircleOutline, GraphicEq, Visibility } from "@mui/icons-material";
import { useNavigate } from "react-router-dom";

export default function HomePage() {
  const navigate = useNavigate();
  
  // 애니메이션 시퀀스를 위한 상태
  const [showHeadline, setShowHeadline] = useState(false);
  const [showSubtitle, setShowSubtitle] = useState(false);
  const [showCards, setShowCards] = useState(false);
  const [showButtons, setShowButtons] = useState(false);

  // 사용자 설정에 따라 애니메이션 활성화 여부 확인
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);

  useEffect(() => {
    // 사용자가 모션 감소를 선호하는지 확인
    const mediaQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    setPrefersReducedMotion(mediaQuery.matches);

    // 설정 변경 감지
    const handleChange = () => setPrefersReducedMotion(mediaQuery.matches);
    mediaQuery.addEventListener('change', handleChange);
    return () => mediaQuery.removeEventListener('change', handleChange);
  }, []);

  // 애니메이션 순서 제어
  useEffect(() => {
    if (prefersReducedMotion) {
      // 접근성을 위해 애니메이션 없이 모든 요소 표시
      setShowHeadline(true);
      setShowSubtitle(true);
      setShowCards(true);
      setShowButtons(true);
    } else {
      // 순차적 애니메이션 적용
      const timer1 = setTimeout(() => setShowHeadline(true), 300);
      const timer2 = setTimeout(() => setShowSubtitle(true), 700);
      const timer4 = setTimeout(() => setShowCards(true), 1200);
      const timer5 = setTimeout(() => setShowButtons(true), 1600);

      return () => {
        clearTimeout(timer1);
        clearTimeout(timer2);
        clearTimeout(timer4);
        clearTimeout(timer5);
      };
    }
  }, [prefersReducedMotion]);

  // 카드 호버 상태 관리
  const [hoveredCard, setHoveredCard] = useState<number | null>(null);

  const handleCardHover = (index: number) => {
    setHoveredCard(index);
  };

  const handleCardLeave = () => {
    setHoveredCard(null);
  };

  // 기능 카드 데이터
  const featureCards = [
    {
      icon: <PlayCircleOutline fontSize="large" sx={{ color: '#333' }} />,
      title: "말하기 속도 분석",
      description: "분당 단어 수를 기반으로 말의 빠르기를 분석해 드려요."
    },
    {
      icon: <GraphicEq fontSize="large" sx={{ color: '#333' }} />,
      title: "음량 안정성 체크",
      description: "안정적인 목소리를 위해 음량의 편차를 분석해 드려요."
    },
    {
      icon: <Visibility fontSize="large" sx={{ color: '#333' }} />,
      title: "비언어 표현 분석",
      description: "자세, 시선, 표정 등 비언어적 요소를 인공지능이 분석해요."
    }
  ];

  return (
    <Box
      component="main"
      sx={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        py: { xs: 6, md: 8 },
        background: "#FFFFFF",
      }}
    >
      <Container maxWidth="lg">
        {/* 로고 및 브랜드 요소 */}
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'center',
            mb: 4
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

        {/* 헤드라인 - h1 사용으로 SEO 최적화 */}
        <Fade in={showHeadline || prefersReducedMotion} timeout={1000}>
          <Typography 
            variant="h1" 
            fontWeight={800} 
            textAlign="center" 
            mb={3}
            sx={{
              color: '#000',
              fontSize: { xs: '2rem', sm: '2.5rem', md: '3.2rem' },
              position: 'relative'
            }}
          >
            당신의 발표, 과연 어땠을까요?
          </Typography>
        </Fade>

        <Fade in={showSubtitle || prefersReducedMotion} timeout={1000}>
          <Typography 
            variant="h2" 
            component="h2"
            color="text.secondary" 
            textAlign="center" 
            mb={4}
            sx={{ 
              maxWidth: '800px',
              mx: 'auto',
              fontWeight: 400,
              fontSize: { xs: '1rem', md: '1.2rem' }
            }}
          >
            발표 영상을 업로드하면, 발표력 분석 결과를 바로 알려드려요.
          </Typography>
        </Fade>

        {/* 주요 기능 카드 */}
        <Grow in={showCards || prefersReducedMotion} timeout={1000}>
          <Grid container spacing={4} justifyContent="center" sx={{ maxWidth: '1200px', mb: 6 }}>
            {featureCards.map((card, index) => (
              <Grid item xs={12} sm={6} md={4} key={index}>
                <Card 
                  sx={{ 
                    borderRadius: 2, 
                    boxShadow: hoveredCard === index 
                      ? '0 14px 28px rgba(0,0,0,0.15), 0 10px 10px rgba(0,0,0,0.12)'
                      : '0 4px 8px rgba(0,0,0,0.08)',
                    height: '100%',
                    transition: 'all 0.3s ease-in-out',
                    transform: hoveredCard === index ? 'translateY(-8px)' : 'none',
                    bgcolor: hoveredCard === index ? '#ffffff' : '#f8f8f8',
                    overflow: 'hidden',
                    cursor: 'pointer',
                    position: 'relative',
                    '&::after': {
                      content: '""',
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      width: '100%',
                      height: '4px',
                      background: '#000',
                      opacity: hoveredCard === index ? 1 : 0,
                      transition: 'opacity 0.3s ease'
                    }
                  }}
                  onMouseEnter={() => handleCardHover(index)}
                  onMouseLeave={handleCardLeave}
                  tabIndex={0}
                  role="button"
                  aria-pressed={hoveredCard === index}
                >
                  <CardContent sx={{ 
                    p: 4, 
                    textAlign: "center",
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'center'
                  }}>
                    <Box 
                      sx={{ 
                        mb: 2,
                        display: 'flex',
                        justifyContent: 'center',
                        alignItems: 'center',
                        width: '70px',
                        height: '70px',
                        borderRadius: '50%',
                        background: 'rgba(0, 0, 0, 0.05)',
                        mx: 'auto',
                        transition: 'all 0.3s ease',
                        transform: hoveredCard === index ? 'scale(1.1)' : 'scale(1)'
                      }}
                    >
                      {card.icon}
                    </Box>
                    <Typography 
                      variant="h6" 
                      component="h4"
                      fontWeight={600}
                      sx={{ 
                        mb: 2,
                        fontSize: '1.25rem'
                      }}
                    >
                      {card.title}
                    </Typography>
                    <Typography 
                      variant="body2" 
                      color="text.secondary"
                      sx={{ 
                        // 모바일에서도 표시되도록 변경, 기본적으로 표시하고 호버 시 강조
                        opacity: hoveredCard === index ? 1 : 0.8,
                        maxHeight: '100px',
                        transition: 'all 0.3s ease',
                      }}
                    >
                      {card.description}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Grow>

        {/* CTA 버튼 */}
        <Fade in={showButtons || prefersReducedMotion} timeout={1000}>
          <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
            <Button
              size="large"
              variant="contained"
              sx={{ 
                px: { xs: 4, md: 6 }, 
                py: { xs: 1.5, md: 2 }, 
                borderRadius: 5, 
                fontWeight: 700, 
                fontSize: { xs: '1rem', md: '1.2rem' }, 
                mb: 2,
                backgroundColor: '#000',
                color: '#fff',
                boxShadow: '0 4px 20px rgba(0, 0, 0, 0.15)',
                transition: 'all 0.3s ease',
                '&:hover': {
                  boxShadow: '0 6px 25px rgba(0, 0, 0, 0.25)',
                  transform: 'translateY(-3px)',
                  backgroundColor: '#333'
                }
              }}
              onClick={() => navigate("/signin")}
            >
              내 발표 분석 시작하기
            </Button>
          </Box>
        </Fade>

        {/* 푸터 - 저작권 및 브랜드 정보 */}
        <Box 
          component="footer"
          sx={{ 
            mt: 12, 
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
}