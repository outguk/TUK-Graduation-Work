import React, { useState } from 'react';
import {
  Box,
  Button,
  Typography,
  Container,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Paper,
  Stack,
  Card,
  CardContent,
} from '@mui/material';
import axios from 'axios';

// 분석 결과 타입 정의
type AnalysisResult = {
  _id: string;
  user_id: number;
  filename: string;
  speaking_speed?: any;
  volume_analysis?: any;
  speaking_evaluation?: any;
  volume_evaluation?: any;
  nonverbal_analysis?: {
    [key: string]: {
      sample_number: number;
      time_range: string;
      is_normal: boolean;
      top_classes: { class: string; probability: number }[];
    };
  };
  message?: string;
};

const UploadPage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // 파일 업로드
  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    setLoading(true);
    setError(null);

    try {
      // '/upload' → Spring → FastAPI
      const response = await axios.post('/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      console.log("📦 분석 응답:", response.data);
      // 방금 업로드한 영상 결과만 result에 저장
      setResult(response.data);
    } catch (e: any) {
      setError("업로드 또는 분석 중 오류가 발생했습니다.");
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="lg">
      <Box component="main" mt={6} mb={5} p={4} borderRadius={3} boxShadow={4} bgcolor="background.paper">
        <Typography variant="h4" gutterBottom align="center" color="primary">
          발표 영상 업로드
        </Typography>
        <Stack spacing={3} alignItems="center">
          <Box width="100%">
            <Typography variant="subtitle1" gutterBottom>
              MP4 파일 선택
            </Typography>
            <input
              id="video-upload"
              type="file"
              accept="video/mp4"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              style={{ display: 'block', marginBottom: 8 }}
            />
            <Typography variant="body2" color="text.secondary">
              최대 500MB까지 업로드 가능합니다.
            </Typography>
          </Box>
          <Button
            variant="contained"
            color="primary"
            onClick={handleUpload}
            disabled={!file || loading}
            size="large"
            sx={{ px: 6, py: 1.5, borderRadius: 2 }}
          >
            업로드
          </Button>
        </Stack>

        {loading && (
          <Box mt={3} display="flex" justifyContent="center">
            <CircularProgress />
          </Box>
        )}

        {error && (
          <Typography color="error" mt={3} align="center">
            {error}
          </Typography>
        )}
      </Box>

      {/* 업로드가 완료되면 result에 단건 결과 표시 */}
      {result && (
        <Box mt={6}>
          <Typography variant="h5" gutterBottom align="center">
            분석 결과 (최근 업로드 파일)
          </Typography>

          <Box mt={3} sx={{ display: 'flex', gap: 3, overflowX: 'auto', pb: 2 }}>
            <Card sx={{ minWidth: 320, borderRadius: 3, boxShadow: 3 }}>
              <CardContent>
                <Typography variant="subtitle1" gutterBottom>🗣 말하기 속도 분석</Typography>
                <pre style={{ whiteSpace: 'pre-wrap', wordWrap: 'break-word', fontFamily: 'monospace', fontSize: '0.9rem' }}>
                  {JSON.stringify(result.speaking_speed ?? '없음', null, 2)}
                </pre>
              </CardContent>
            </Card>

            <Card sx={{ minWidth: 320, borderRadius: 3, boxShadow: 3 }}>
              <CardContent>
                <Typography variant="subtitle1" gutterBottom>🔊 음량 분석</Typography>
                <pre style={{ whiteSpace: 'pre-wrap', wordWrap: 'break-word', fontFamily: 'monospace', fontSize: '0.9rem' }}>
                  {JSON.stringify(result.volume_analysis ?? '없음', null, 2)}
                </pre>
              </CardContent>
            </Card>

            <Card sx={{ minWidth: 320, borderRadius: 3, boxShadow: 3 }}>
              <CardContent>
                <Typography variant="subtitle1" gutterBottom>📊 점수 요약</Typography>
                <Typography variant="body2" fontWeight={500}>🏃 속도 점수</Typography>
                <pre style={{ fontFamily: 'monospace' }}>{JSON.stringify(result.speaking_evaluation ?? '없음', null, 2)}</pre>
                <Typography variant="body2" fontWeight={500}>🔈 음량 점수</Typography>
                <pre style={{ fontFamily: 'monospace' }}>{JSON.stringify(result.volume_evaluation ?? '없음', null, 2)}</pre>
              </CardContent>
            </Card>
          </Box>

          <Box mt={5}>
            <Typography variant="h6" gutterBottom>🧍 비언어적 분석 상세</Typography>
            {result.nonverbal_analysis ? (
              <Paper elevation={3} sx={{ borderRadius: 2 }}>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell>샘플 번호</TableCell>
                      <TableCell>시간 구간</TableCell>
                      <TableCell>정상 여부</TableCell>
                      <TableCell>상위 행동 분석</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {Object.entries(result.nonverbal_analysis).map(([key, item]) => (
                      <TableRow key={key}>
                        <TableCell>{item.sample_number}</TableCell>
                        <TableCell>{item.time_range}</TableCell>
                        <TableCell>{item.is_normal ? '정상' : '비정상'}</TableCell>
                        <TableCell>
                          {!item.is_normal ? (
                            <ul style={{ paddingLeft: 16 }}>
                              {item.top_classes.map((cls, i) => (
                                <li key={i}>
                                  {cls.class} ({cls.probability}%)
                                </li>
                              ))}
                            </ul>
                          ) : (
                            '-'
                          )}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </Paper>
            ) : (
              <Typography>비언어 분석 결과 없음</Typography>
            )}
          </Box>
        </Box>
      )}
    </Container>
  );
};

export default UploadPage;