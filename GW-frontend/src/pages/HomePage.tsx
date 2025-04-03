import Grid from '@mui/material/Grid';
import { Box, Typography, Button, Card, CardContent } from "@mui/material";
import { PlayCircleOutline, GraphicEq, Visibility } from "@mui/icons-material";
import { useNavigate } from "react-router-dom";

export default function HomePage() {
  const navigate = useNavigate();

  return (
    <Box
      sx={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        px: 3,
        background: "linear-gradient(to right, #f2f6ff, #e3ecff)",
      }}
    >
      {/* 헤드라인 */}
      <Typography variant="h3" fontWeight={700} textAlign="center" mb={2}>
        당신의 발표, 과연 어땠을까요?
      </Typography>
      <Typography variant="h6" color="text.secondary" textAlign="center" mb={5}>
        발표 영상을 업로드하면, 발표력 분석 결과를 바로 알려드립니다.
      </Typography>

      {/* 주요 기능 카드 */}
      <Grid container spacing={3} justifyContent="center">
        {/* 말하기 속도 분석 */}
        <Grid item xs={12} md={4}>
          <Card sx={{ p: 3, borderRadius: 4, boxShadow: 4 }}>
            <CardContent sx={{ textAlign: "center" }}>
              <PlayCircleOutline fontSize="large" color="primary" />
              <Typography variant="h6" mt={2}>말하기 속도 분석</Typography>
              <Typography variant="body2" color="text.secondary">
                분당 단어 수를 기반으로 말의 빠르기를 분석해 드려요.
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* 음량 안정성 체크 */}
        <Grid item xs={12} md={4}>
          <Card sx={{ p: 3, borderRadius: 4, boxShadow: 4 }}>
            <CardContent sx={{ textAlign: "center" }}>
              <GraphicEq fontSize="large" color="primary" />
              <Typography variant="h6" mt={2}>음량 안정성 체크</Typography>
              <Typography variant="body2" color="text.secondary">
                안정적인 목소리를 위해 음량의 편차를 분석해 드려요.
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* 비언어 표현 분석 */}
        <Grid item xs={12} md={4}>
          <Card sx={{ p: 3, borderRadius: 4, boxShadow: 4 }}>
            <CardContent sx={{ textAlign: "center" }}>
              <Visibility fontSize="large" color="primary" />
              <Typography variant="h6" mt={2}>비언어 표현 분석</Typography>
              <Typography variant="body2" color="text.secondary">
                자세, 시선, 표정 등 비언어적 요소를 인공지능이 분석해요.
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* CTA 버튼 */}
      <Button
        size="large"
        variant="contained"
        color="primary"
        sx={{ px: 6, py: 2, borderRadius: 10, fontWeight: 600, fontSize: "1.2rem", mt: 5 }}
        onClick={() => navigate("/signin")}
      >
        내 발표 분석 시작하기
      </Button>

      <Button
        variant="text"
        sx={{ mt: 2, textDecoration: "underline" }}
        onClick={() => navigate("/sample-result")}
      >
        샘플 결과 먼저 보기
      </Button>
    </Box>
  );
}
