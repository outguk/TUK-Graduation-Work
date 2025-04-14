/**
 * PresentationSidebar.tsx
 * 
 * 백엔드 개발자를 위한 안내:
 * 이 파일은 발표 분석 대시보드의 좌측 사이드바 컴포넌트입니다.
 * 사용자 정보와 발표 목록을 표시하며, 각 발표를 선택할 수 있는 UI를 제공합니다.
 * 
 * 필요한 백엔드 엔드포인트:
 * 1. GET /api/presentations - 사용자의 모든 발표 목록
 * 2. GET /api/user/profile - 현재 로그인한 사용자 정보
 */

import React from 'react';
import {
  Box,
  Typography,
  List,
  ListItemButton,
  ListItemText,
  Avatar,
  Divider,
  Paper,
} from '@mui/material';
import { AccessTime as AccessTimeIcon } from '@mui/icons-material';

/**
 * 백엔드 개발자 참고사항:
 * 사이드바에 필요한 데이터 타입 정의입니다.
 * API 응답 형식 설계 시 참고하세요.
 */

/**
 * 발표 항목 타입
 * 
 * 백엔드 개발 참고사항:
 * - GET /api/presentations 엔드포인트의 응답 항목 형식입니다.
 * - 모든 필드는 필수입니다.
 */
interface PresentationItem {
  id: string;        // 발표 고유 식별자 (UUID 또는 기타 고유 문자열)
  title: string;     // 발표 제목
  date: string;      // 발표 날짜 (YYYY.MM.DD 형식 권장)
  duration: string;  // 발표 길이 (M:SS 형식, 예: "0:45")
}

/**
 * 사용자 프로필 타입
 * 
 * 백엔드 개발 참고사항:
 * - GET /api/user/profile 엔드포인트의 응답에서 필요한 부분입니다.
 * - 현재는 이름만 사용하고 있지만, 추가 정보를 포함할 수 있습니다.
 */
interface UserProfile {
  name: string;      // 사용자 이름
  // 향후 사용자 기타 정보를 추가할 수 있습니다.
}

/**
 * PresentationSidebar Props 타입
 * 
 * 백엔드 개발 참고사항:
 * - 이 컴포넌트가 상위 컴포넌트에서 받는 props 목록입니다.
 * - 실제 구현 시 API 응답을 이 형식에 맞게 가공하여 전달해야 합니다.
 */
interface PresentationSidebarProps {
  presentations: PresentationItem[];        // 발표 목록
  selectedPresentationId: string | null;    // 현재 선택된 발표 ID
  onSelectPresentation: (id: string) => void; // 발표 선택 이벤트 핸들러
  userProfile: UserProfile;                // 사용자 프로필 정보
}

/**
 * 발표 분석 사이드바 컴포넌트
 * 
 * 백엔드 개발 참고사항:
 * - 사용자 정보와 발표 목록을 표시합니다.
 * - 각 발표를 클릭하면 해당 발표의 분석 결과로 이동합니다.
 * - 데이터는 상위 컴포넌트(AnalysisDashboardPage)에서 props로 전달받습니다.
 */
const PresentationSidebar: React.FC<PresentationSidebarProps> = ({
  presentations,
  selectedPresentationId,
  onSelectPresentation,
  userProfile
}) => {
  // 사용자 이름의 첫 글자를 아바타로 사용
  const getInitial = (name: string) => {
    return name ? name.charAt(0).toUpperCase() : 'U';
  };

  return (
    <Box
      sx={{
        width: '8%',
        minWidth: '200px',
        height: '100vh',
        position: 'fixed',
        borderRight: '1px solid #eee',
        backgroundColor: '#fbfbfc',
        display: 'flex',
        flexDirection: 'column',
        padding: '2rem 0',
        overflow: 'auto',
      }}
    >
      {/* 사용자 프로필 섹션 */}
      <Box sx={{ display: 'flex', alignItems: 'center', px: 3, mb: 4 }}>
        {/* 
         * 백엔드 개발 참고사항:
         * 여기서는 사용자 이름의 첫 글자로 아바타를 표시하고 있습니다.
         * GET /api/user/profile에서 profile_image_url과 같은 필드를 제공하면
         * 실제 사용자 이미지로 대체할 수 있습니다.
         */}
        <Avatar
          sx={{
            width: 40,
            height: 40,
            bgcolor: '#000',
            fontSize: '1.5rem',
            fontWeight: 500,
            mr: 2
          }}
        >
          {getInitial(userProfile.name)}
        </Avatar>
        <Typography variant="subtitle1" fontWeight={600}>
          {userProfile.name}
        </Typography>
      </Box>

      <Divider sx={{ mb: 3 }} />

      {/* 
       * 백엔드 개발 참고사항:
       * 발표 목록 섹션입니다.
       * GET /api/presentations 응답의 항목들이 여기에 표시됩니다.
       * 각 항목은 PresentationItem 인터페이스를 따라야 합니다.
       */}
      <Typography variant="subtitle2" sx={{ px: 3, mb: 2, color: '#666' }}>
        내 발표 분석
      </Typography>

      <List sx={{ px: 2 }}>
        {presentations.map((presentation) => (
          <Paper
            key={presentation.id}
            elevation={0}
            sx={{
              mb: 1.5,
              borderRadius: 2,
              overflow: 'hidden',
              backgroundColor: selectedPresentationId === presentation.id ? '#fff' : 'transparent',
              boxShadow: selectedPresentationId === presentation.id ? '0 4px 12px rgba(0,0,0,0.05)' : 'none',
              border: '1px solid',
              borderColor: selectedPresentationId === presentation.id ? '#fff' : 'transparent',
              transition: 'all 0.2s ease',
            }}
          >
            <ListItemButton
              selected={selectedPresentationId === presentation.id}
              onClick={() => onSelectPresentation(presentation.id)}
              sx={{
                borderRadius: 2,
                py: 1.5,
                '&.Mui-selected': {
                  backgroundColor: 'transparent',
                  '&:hover': {
                    backgroundColor: 'rgba(0,0,0,0.02)',
                  },
                },
              }}
            >
              <ListItemText
                primary={presentation.title}
                secondary={
                  <Box sx={{ display: 'flex', alignItems: 'center', mt: 0.5 }}>
                    <Typography variant="caption" color="text.secondary">
                      {presentation.date}
                    </Typography>
                    <Box sx={{ display: 'flex', alignItems: 'center', ml: 2 }}>
                      <AccessTimeIcon sx={{ fontSize: 12, mr: 0.5, color: 'text.secondary' }} />
                      <Typography variant="caption" color="text.secondary">
                        {presentation.duration}
                      </Typography>
                    </Box>
                  </Box>
                }
                primaryTypographyProps={{
                  variant: 'body2',
                  fontWeight: selectedPresentationId === presentation.id ? 600 : 400,
                  sx: {
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  },
                }}
                sx={{ m: 0 }}
              />
            </ListItemButton>
          </Paper>
        ))}
      </List>
    </Box>
  );
};

/**
 * 백엔드 개발자를 위한 종합 API 명세
 * 
 * 1. GET /api/presentations
 *    - 설명: 사용자의 모든 발표 목록 조회
 *    - 응답: PresentationItem[] 형식
 *    - 예시:
 *      [
 *        {
 *          "id": "pres-001",
 *          "title": "팀 프로젝트 발표",
 *          "date": "2025.04.10",
 *          "duration": "0:45"
 *        },
 *        {
 *          "id": "pres-002",
 *          "title": "취업 인터뷰 연습",
 *          "date": "2025.04.03",
 *          "duration": "0:52"
 *        }
 *      ]
 * 
 * 2. GET /api/user/profile
 *    - 설명: 현재 로그인한 사용자 정보 조회
 *    - 응답: { name: string, ... } 형식
 *    - 예시: 
 *      {
 *        "name": "홍길동",
 *        "email": "user@example.com",
 *        // 추가 정보 (선택적)
 *        "profile_image_url": "https://example.com/images/profile.jpg" 
 *      }
 * 
 * 3. 데이터 흐름
 *    - AnalysisDashboardPage에서 API 호출 후 PresentationSidebar로 데이터 전달
 *    - 사용자가 발표를 선택하면 onSelectPresentation 콜백을 통해 AnalysisDashboardPage에 알림
 *    - AnalysisDashboardPage는 선택된 발표 ID에 해당하는 상세 분석 데이터를 로드
 */

export default PresentationSidebar;