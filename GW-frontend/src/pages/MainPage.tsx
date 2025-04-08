/**
 * MainPage.tsx
 * 
 * 사용자가 로그인 후 맨 처음 보는 메인 페이지 컴포넌트입니다.
 * 이 페이지에서는 주요 서비스 기능에 접근할 수 있는 카드형 메뉴를 제공합니다.
 * 
 * 주요 기능:
 * - 사용자 개인화 환영 메시지 표시
 * - 주요 서비스 기능(발표 분석, 내 정보 등)으로 연결되는 카드 메뉴
 * - 카드 호버 효과 및 반응형 레이아웃
 */

import { useState } from 'react';
import { 
  Box, 
  Typography, 
  Container, 
  Grid, 
  Card, 
  CardContent, 
} from "@mui/material";
import PersonIcon from '@mui/icons-material/Person';
import { useNavigate } from "react-router-dom";

export default function MainPage() {
  const navigate = useNavigate();
  
  // 카드 호버 상태 관리
  // hoveredCard: 현재 마우스가 올라간 카드의 인덱스를 저장 (null: 호버된 카드 없음)
  const [hoveredCard, setHoveredCard] = useState<number | null>(null);

  /**
   * 카드에 마우스를 올렸을 때 호출되는 함수
   * @param index - 호버된 카드의 인덱스
   * 
   * 백엔드 통합 불필요: UI 상태 관리용 함수
   */
  const handleCardHover = (index: number) => {
    setHoveredCard(index);
  };

  /**
   * 카드에서 마우스가 벗어났을 때 호출되는 함수
   * 
   * 백엔드 통합 불필요: UI 상태 관리용 함수
   */
  const handleCardLeave = () => {
    setHoveredCard(null);
  };

  /**
   * 키보드 접근성을 위한 키 이벤트 핸들러
   * Enter 또는 Space 키를 누르면 해당 경로로 이동합니다.
   * 
   * @param event - 키보드 이벤트
   * @param path - 이동할 경로
   * 
   * 백엔드 통합 불필요: UI 접근성 관련 함수
   */
  const handleCardKeyDown = (event: React.KeyboardEvent, path: string) => {
    if (event.key === 'Enter' || event.key === ' ') {
      navigate(path);
    }
  };

  /**
   * 서비스 카드 데이터 배열
   * 
   * 백엔드 통합 포인트:
   * - 이 부분은 나중에 API에서 가져온 데이터로 대체될 수 있습니다.
   * - 사용자의 권한이나 서비스 가입 상태에 따라 보여지는 카드를 다르게 설정할 수 있습니다.
   * 
   * 각 카드 객체 속성:
   * - title: 카드에 표시될 제목
   * - description: 카드에 표시될 설명
   * - link: 카드 클릭 시 이동할 경로
   */
  const serviceCards = [
    {
      title: "발표 영상 분석",
      description: "발표 영상 분석을 통한 발표 피드백",
      link: "/upload"
    },
    {
      title: "내 정보",
      description: "내 정보 수정",
      link: "/profile"
    },
    {
      title: "지난 발표 분석",
      description: "지금까지 분석한 발표들",
      link: "/analysis"
    },
    {
      title: "가이드",
      description: "초보자를 위한 발표 영상 분석 가이드",
      link: "/upload"
    }
  ];
  
  /**
   * 그리드 레이아웃 계산 함수
   * 카드별로 그리드 크기를 다르게 지정하여 레이아웃을 구성합니다.
   * 
   * @param index - 카드의 인덱스
   * @returns 해당 인덱스의 카드에 적용할 그리드 사이즈 설정
   * 
   * 백엔드 통합 불필요: UI 레이아웃 관련 함수
   */
  const getGridSize = (index: number) => {
    if (index === 0) {
      return { xs: 12, sm: 8, md: 8, lg: 8 }; // 첫 번째 카드: 큰 사이즈 (중요 기능)
    } else if (index === 1) {
      return { xs: 12, sm: 4, md: 4, lg: 4 }; // 두 번째 카드: 작은 사이즈
    } else if (index === 2) {
      return { xs: 12, sm: 8, md: 8, lg: 8 }; // 세 번째 카드: 큰 사이즈 (중요 기능)
    } else if (index === 3) {
      return { xs: 12, sm: 4, md: 4, lg: 4 }; // 네 번째 카드: 작은 사이즈
    }
    return { xs: 12, sm: 6, md: 6, lg: 6 }; // 기본 사이즈 (추가 카드용)
  };

  return (
    <Box
      component="main"
      sx={{
        minHeight: "100vh",
        width: "100%",
        py: { xs: 4, md: 6 },
        px: { xs: 2, md: 4 },
        background: "#FFFFFF",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <Container maxWidth="xl" sx={{ my: 4 }}>
        {/* 상단 타이틀 및 사용자 인사말 */}
        <Box sx={{ mb: 4 }}>
          <Typography 
            variant="h3" 
            component="h1"
            sx={{ 
              color: '#000',
              fontWeight: 600,
              mb: 3
            }}
          >
            PRESENT INSIGHT
          </Typography>
          
          {/* 
            백엔드 통합 포인트:
            - 로그인한 사용자 이름을 표시하는 부분입니다.
            - API에서 사용자 정보를 가져와 "(사용자)" 부분을 실제 사용자 이름으로 대체해야 합니다.
            - 예: `안녕하세요! ${userInfo.name}`
          */}
          <Typography 
            variant="body1" 
            sx={{ 
              color: 'text.secondary',
              fontWeight: 300,
              mb: 3
            }}
          >
            안녕하세요! (사용자)
          </Typography>
        </Box>

        {/* 서비스 카드 그리드 레이아웃 */}
        <Grid container spacing={4}>
          {/* 
            서비스 카드 매핑
            
            백엔드 통합 포인트:
            - serviceCards 배열이 API에서 가져온 데이터로 대체되면 자동으로 여기에 반영됩니다.
            - 사용자별 맞춤 카드나 동적 컨텐츠가 필요하면 이 부분을 수정하세요.
          */}
          {serviceCards.map((card, index) => {
            const gridSize = getGridSize(index);
            return (
              <Grid 
                item 
                xs={gridSize.xs}
                sm={gridSize.sm}
                md={gridSize.md}
                lg={gridSize.lg}
                key={index}
                sx={{ 
                  // 모바일에서는 순서대로, 태블릿 이상에서는 지그재그 레이아웃 구성
                  order: {
                    xs: 0, // 모바일에서는 순서 변경 없음
                    sm: index === 2 ? 3 : (index === 3 ? 2 : index) // 태블릿 이상에서 3번과 4번 카드의 순서 변경
                  }
                }}
              >
                <Card 
                  role="button"
                  tabIndex={0}
                  aria-label={`Service: ${card.title}`}
                  sx={{ 
                    height: { xs: 280, sm: 300, md: 320 }, // 반응형 높이 설정
                    width: "100%", 
                    borderRadius: 2,
                    backgroundColor: '#f5f5f5',
                    boxShadow: 'none',
                    transition: 'all 0.3s ease',
                    transform: hoveredCard === index ? 'translateY(-4px)' : 'none',
                    '&:hover': {
                      cursor: 'pointer',
                      boxShadow: '0 8px 16px rgba(0,0,0,0.1)',
                    },
                    py: 2,
                    mx: { xs: 0, sm: 1 } 
                  }}
                  onMouseEnter={() => handleCardHover(index)}
                  onMouseLeave={handleCardLeave}
                  onClick={() => navigate(card.link)}
                  onKeyDown={(e) => handleCardKeyDown(e, card.link)}
                >
                  <CardContent sx={{ 
                    p: { xs: 3, sm: 4, md: 5 }, 
                    height: '100%',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'space-between', // 컨텐츠를 위아래로 분산
                  }}>
                    {/* 아이콘을 왼쪽 위에 배치 */}
                    <Box 
                      sx={{ 
                        width: '100%',
                        display: 'flex',
                        justifyContent: 'flex-start',
                        alignItems: 'flex-start'
                      }}
                    >
                      {/* 
                        백엔드 통합 포인트 (향후 확장):
                        - 각 카드별로 다른 아이콘을 표시하려면 serviceCards 배열에 icon 속성을 추가하고
                          여기에서 해당 아이콘을 렌더링하도록 수정하세요.
                        - 현재는 모든 카드에 동일한 PersonIcon이 사용됩니다.
                      */}
                      <PersonIcon sx={{ fontSize: 40, color: '#000' }} />
                    </Box>
                    
                    {/* 카드 텍스트 영역 (제목 및 설명) */}
                    <Box 
                      sx={{ 
                        width: '100%',
                        display: 'flex',
                        flexDirection: 'column',
                        justifyContent: 'flex-end',
                        alignItems: 'flex-start',
                        mt: 'auto' // 아래쪽으로 밀어내기
                      }}
                    >
                      {/* 카드 제목 */}
                      <Typography 
                        variant="h5" 
                        component="h2"
                        fontWeight={500}
                        sx={{ mb: 1.5, textAlign: 'left' }}
                      >
                        {card.title}
                      </Typography>
                      
                      {/* 카드 설명 */}
                      <Typography 
                        variant="body1" 
                        color="text.secondary"
                        sx={{ mb: 2, textAlign: 'left' }}
                      >
                        {card.description}
                      </Typography>
                    </Box>
                  </CardContent>
                </Card>
              </Grid>
            );
          })}
        </Grid>
      </Container>
    </Box>
  );
}