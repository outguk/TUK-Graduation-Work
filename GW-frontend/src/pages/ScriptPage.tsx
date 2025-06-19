import { useEffect, useState } from 'react';
import axios from 'axios';
import { 
  Box, 
  Typography, 
  List, 
  ListItem, 
  ListItemText, 
  Card, 
  CardContent, 
  Grid,
  Paper,
  Divider,
  Alert,
  CircularProgress,
  Chip
} from '@mui/material';

interface ScriptAnalysis {
  length: number;
  min_length: number;
  max_length: number;
  length_feedback: string;
  uncertainty_examples: [number, string][];
  non_honorific_examples: [number, string][];
  subject_verb_examples: [number, string][];
  profanity_examples: [number, string][];
  otas_detected: [number, string, number, string][];
  word_repeat_counter: Record<string, number>;
  all_words: string[];
  uncertainty_count: number;
  non_honorific_count: number;
  subject_verb_mismatch_count: number;
  profanity_count: number;
}

interface ScriptData {
  _id: string;
  script_text: string;
  script_analysis: ScriptAnalysis;
  speech_minutes: number;
  filename: string;
  timestamp: string;
}

interface ScriptPageProps {
  scriptId: string;
}

const ScriptPage: React.FC<ScriptPageProps> = ({ scriptId }) => {
  const [scriptData, setScriptData] = useState<ScriptData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAnalysis = async () => {
      const token = localStorage.getItem('token');
      if (!token) {
        setError('로그인이 필요합니다');
        setLoading(false);
        return;
      }

      try {
        console.log('대본 분석 조회 시작, script_id:', scriptId);
        
        const response = await axios.get('/spring/api/get-script-analysis', {
          params: { script_id: scriptId },
          headers: { Authorization: `Bearer ${token}` },
        });
        
        console.log('대본 분석 응답:', response.data);
        
        if (response.data && response.data.script_analysis) {
          setScriptData(response.data);
          setError(null);
        } else {
          setError('분석 결과가 없습니다');
        }
      } catch (error: any) {
        console.error('대본 분석 조회 실패:', error);
        if (error.response) {
          setError(`분석 결과 조회 실패: ${error.response.data.message || error.response.data.error || error.message}`);
        } else {
          setError('분석 결과 조회 실패: ' + error.message);
        }
      } finally {
        setLoading(false);
      }
    };

    if (scriptId) {
      fetchAnalysis();
    }
  }, [scriptId]);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ m: 3 }}>
        {error}
      </Alert>
    );
  }

  if (!scriptData || !scriptData.script_analysis) {
    return (
      <Alert severity="warning" sx={{ m: 3 }}>
        분석 데이터를 불러올 수 없습니다
      </Alert>
    );
  }

  const analysis = scriptData.script_analysis;

  return (
    <Box sx={{ padding: 3 }}>
      {/* 헤더 정보 */}
      <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
        <Typography variant="h4" gutterBottom fontWeight="bold">
          대본 분석 결과
        </Typography>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="body2" color="text.secondary">파일명</Typography>
            <Typography variant="h6">{scriptData.filename}</Typography>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="body2" color="text.secondary">발표 예정 시간</Typography>
            <Typography variant="h6">{scriptData.speech_minutes}분</Typography>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="body2" color="text.secondary">분석 일시</Typography>
            <Typography variant="h6">
              {new Date(scriptData.timestamp).toLocaleDateString('ko-KR')}
            </Typography>
          </Grid>
          <Grid item xs={12} sm={6} md={3}>
            <Typography variant="body2" color="text.secondary">총 글자수</Typography>
            <Typography variant="h6">{analysis.length}자</Typography>
          </Grid>
        </Grid>
      </Paper>

      {/* 글자수 분석 */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            📝 글자수 분석
          </Typography>
          <Box sx={{ mb: 2 }}>
            <Typography variant="body1">
              <strong>현재 글자수:</strong> {analysis.length}자
            </Typography>
            <Typography variant="body1">
              <strong>권장 범위:</strong> {analysis.min_length}자 ~ {analysis.max_length}자
            </Typography>
          </Box>
          <Alert 
            severity={
              analysis.length >= analysis.min_length && analysis.length <= analysis.max_length 
                ? "success" 
                : "warning"
            }
          >
            {analysis.length_feedback}
          </Alert>
        </CardContent>
      </Card>

      {/* 문제점 요약 */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h5" gutterBottom>
            🔍 문제점 요약
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={6} sm={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h4" color="error">
                  {analysis.uncertainty_count}
                </Typography>
                <Typography variant="body2">불확실 표현</Typography>
              </Box>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h4" color="warning.main">
                  {analysis.non_honorific_count}
                </Typography>
                <Typography variant="body2">비격식 표현</Typography>
              </Box>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h4" color="error">
                  {analysis.subject_verb_mismatch_count}
                </Typography>
                <Typography variant="body2">주어-서술어 불일치</Typography>
              </Box>
            </Grid>
            <Grid item xs={6} sm={3}>
              <Box sx={{ textAlign: 'center' }}>
                <Typography variant="h4" color="error">
                  {analysis.profanity_count}
                </Typography>
                <Typography variant="body2">비속어</Typography>
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* 상세 분석 결과 */}
      <Grid container spacing={3}>
        {/* 불확실 표현 */}
        {analysis.uncertainty_count > 0 && (
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom color="error">
                  ❓ 불확실 표현 ({analysis.uncertainty_count}개)
                </Typography>
                <List dense>
                  {analysis.uncertainty_examples.slice(0, 5).map(([pos, text], i) => (
                    <ListItem key={i}>
                      <ListItemText 
                        primary={`${pos}번째 문장`}
                        secondary={`"${text}"`}
                      />
                    </ListItem>
                  ))}
                  {analysis.uncertainty_examples.length > 5 && (
                    <ListItem>
                      <ListItemText 
                        primary={`외 ${analysis.uncertainty_examples.length - 5}개 더...`}
                      />
                    </ListItem>
                  )}
                </List>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* 비격식 표현 */}
        {analysis.non_honorific_count > 0 && (
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom color="warning.main">
                  💬 비격식 표현 ({analysis.non_honorific_count}개)
                </Typography>
                <List dense>
                  {analysis.non_honorific_examples.slice(0, 5).map(([pos, text], i) => (
                    <ListItem key={i}>
                      <ListItemText 
                        primary={`${pos}번째 문장`}
                        secondary={`"${text}"`}
                      />
                    </ListItem>
                  ))}
                  {analysis.non_honorific_examples.length > 5 && (
                    <ListItem>
                      <ListItemText 
                        primary={`외 ${analysis.non_honorific_examples.length - 5}개 더...`}
                      />
                    </ListItem>
                  )}
                </List>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* 주어-서술어 불일치 */}
        {analysis.subject_verb_mismatch_count > 0 && (
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom color="error">
                  ⚠️ 주어-서술어 불일치 ({analysis.subject_verb_mismatch_count}개)
                </Typography>
                <List dense>
                  {analysis.subject_verb_examples.slice(0, 5).map(([pos, text], i) => (
                    <ListItem key={i}>
                      <ListItemText 
                        primary={`${pos}번째 문장`}
                        secondary={`"${text}"`}
                      />
                    </ListItem>
                  ))}
                  {analysis.subject_verb_examples.length > 5 && (
                    <ListItem>
                      <ListItemText 
                        primary={`외 ${analysis.subject_verb_examples.length - 5}개 더...`}
                      />
                    </ListItem>
                  )}
                </List>
              </CardContent>
            </Card>
          </Grid>
        )}

        {/* 오타 감지 */}
        {analysis.otas_detected.length > 0 && (
          <Grid item xs={12} md={6}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom color="info.main">
                  🔍 오타 감지 ({analysis.otas_detected.length}개)
                </Typography>
                <List dense>
                  {analysis.otas_detected.slice(0, 5).map(([pos, char, charIdx, text], i) => (
                    <ListItem key={i}>
                      <ListItemText 
                        primary={`${pos}번째 문장, ${charIdx}번째 문자`}
                        secondary={`"${char}" in "${text}"`}
                      />
                    </ListItem>
                  ))}
                  {analysis.otas_detected.length > 5 && (
                    <ListItem>
                      <ListItemText 
                        primary={`외 ${analysis.otas_detected.length - 5}개 더...`}
                      />
                    </ListItem>
                  )}
                </List>
              </CardContent>
            </Card>
          </Grid>
        )}
      </Grid>

      {/* 단어 반복 빈도 */}
      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            🔄 단어 반복 빈도 (2회 이상)
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {Object.entries(analysis.word_repeat_counter)
              .filter(([_, count]) => count > 1)
              .sort(([,a], [,b]) => b - a)
              .slice(0, 20)
              .map(([word, count], i) => (
                <Chip 
                  key={i} 
                  label={`${word} (${count}회)`} 
                  color={count > 3 ? "warning" : "default"}
                  variant="outlined"
                />
              ))}
          </Box>
        </CardContent>
      </Card>

      {/* 대본 전문 */}
      <Card sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            📄 대본 전문
          </Typography>
          <Divider sx={{ mb: 2 }} />
          <Typography 
            variant="body1" 
            sx={{ 
              whiteSpace: 'pre-wrap', 
              lineHeight: 1.8,
              backgroundColor: '#f5f5f5',
              p: 2,
              borderRadius: 1,
              maxHeight: '400px',
              overflow: 'auto'
            }}
          >
            {scriptData.script_text}
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
};

export default ScriptPage;