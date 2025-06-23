import { useEffect, useState, useMemo, JSX } from 'react';
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
  Badge
} from '@mui/material';
import HelpOutlineIcon from '@mui/icons-material/HelpOutline';
import ChatBubbleOutlineIcon from '@mui/icons-material/ChatBubbleOutline';
import WarningAmberOutlinedIcon from '@mui/icons-material/WarningAmberOutlined';
import BlockOutlinedIcon from '@mui/icons-material/BlockOutlined';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';

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

const HIGHLIGHT_COLORS: Record<string, string> = {
  uncertainty: 'rgba(255, 204, 204, 0.7)', // Light Red
  non_honorific: 'rgba(255, 236, 179, 0.7)', // Light Yellow
  subject_verb: 'rgba(209, 196, 233, 0.7)', // Light Purple
  profanity: 'rgba(255, 209, 128, 0.7)', // Light Orange
};

const ICONS: Record<string, JSX.Element> = {
  uncertainty: <HelpOutlineIcon sx={{ color: '#e57373', mr: 1.5 }} />,
  non_honorific: <ChatBubbleOutlineIcon sx={{ color: '#ffd54f', mr: 1.5 }} />,
  subject_verb: <WarningAmberOutlinedIcon sx={{ color: '#9575cd', mr: 1.5 }} />,
  profanity: <BlockOutlinedIcon sx={{ color: '#ffb74d', mr: 1.5 }} />,
};

const ScriptPage: React.FC<ScriptPageProps> = ({ scriptId }) => {
  const [scriptData, setScriptData] = useState<ScriptData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeHighlightTypes, setActiveHighlightTypes] = useState<Set<string>>(new Set());
  const [hoveredCard, setHoveredCard] = useState<string | null>(null);

  useEffect(() => {
    let isMounted = true;
    
    const fetchAnalysis = async () => {
      const token = localStorage.getItem('token');
      if (!token) {
        if (isMounted) {
          setError('로그인이 필요합니다');
          setLoading(false);
        }
        return;
      }

      try {
        console.log('대본 분석 조회 시작, script_id:', scriptId);
        
        const response = await axios.get('/spring/api/get-script-analysis', {
          params: { script_id: scriptId },
          headers: { Authorization: `Bearer ${token}` },
        });
        
        if (!isMounted) return; // 컴포넌트가 언마운트된 경우 처리 중단
        
        console.log('대본 분석 응답:', response.data);
        
        if (response.data && response.data.script_analysis) {
          setScriptData(response.data);
          setError(null);
        } else {
          setError('분석 결과가 없습니다');
        }
      } catch (error: any) {
        if (!isMounted) return; // 컴포넌트가 언마운트된 경우 처리 중단
        
        console.error('대본 분석 조회 실패:', error);
        if (error.response) {
          setError(`분석 결과 조회 실패: ${error.response.data.message || error.response.data.error || error.message}`);
        } else {
          setError('분석 결과 조회 실패: ' + error.message);
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    };

    if (scriptId) {
      fetchAnalysis();
    }

    return () => {
      isMounted = false;
    };
  }, [scriptId]);

  const handleAnalysisTypeClick = (type: string) => {
    setActiveHighlightTypes(prev => {
      const newTypes = new Set(prev);
      if (newTypes.has(type)) {
        newTypes.delete(type);
      } else {
        newTypes.add(type);
      }
      return newTypes;
    });
  };

  const highlightMap = useMemo<Record<string, Set<number>>>(() => {
    if (!scriptData || activeHighlightTypes.size === 0) {
      return {};
    }

    const analysisData = scriptData.script_analysis;
    const highlightMap: Record<string, Set<number>> = {};
    
    // 문장 개수 계산 (백엔드와 동일하게 \r 제거 후 분할)
    const totalSentences = scriptData.script_text
      .replace(/\r/g, '')
      .split(/[.!?\n]/)
      .map(s => s.trim())
      .filter(s => s.length > 0).length;
    
    activeHighlightTypes.forEach(type => {
      let examples: [number, string][] = [];
      
      switch (type) {
        case 'uncertainty':
          examples = analysisData.uncertainty_examples;
          break;
        case 'non_honorific':
          examples = analysisData.non_honorific_examples;
          break;
        case 'subject_verb':
          examples = analysisData.subject_verb_examples;
          break;
        case 'profanity':
          examples = analysisData.profanity_examples;
          break;
      }
      // 인덱스 번호만 추출 (1부터 시작하는 인덱스를 0부터 시작하는 인덱스로 변환)
      // 범위 검증 추가: 0 <= idx-1 < totalSentences
      highlightMap[type] = new Set(
        examples
          .map(([idx, _]) => idx - 1)
          .filter(adjustedIdx => adjustedIdx >= 0 && adjustedIdx < totalSentences)
      );
      console.log(`${type} 하이라이트할 문장 인덱스들:`, Array.from(highlightMap[type]));
    });
    
    return highlightMap;
  }, [scriptData, activeHighlightTypes]);

  const renderHighlightedScript = () => {
    if (!scriptData) return null;

    // Python과 동일한 방식으로 문장 분할 (정규식 패턴도 동일하게, \r 제거)
    const sentences = scriptData.script_text
      .replace(/\r/g, '')
      .split(/[.!?\n]/)
      .map(s => s.trim())
      .filter(s => s.length > 0);

    let restOfText = scriptData.script_text;
    const renderedElements: (string | JSX.Element)[] = [];

    sentences.forEach((sentence, idx) => {
      // 문장이 원본 텍스트에 존재하는지 확인
      const originalIndex = restOfText.indexOf(sentence);
      
      if (originalIndex === -1) {
        // 문장을 찾을 수 없는 경우 (드문 경우지만 안전장치)
        if(restOfText.length > 0){
          renderedElements.push(restOfText);
          restOfText = "";
        }
        return;
      }

      const precedingText = restOfText.substring(0, originalIndex);
      if (precedingText) {
        renderedElements.push(precedingText);
      }

      // 인덱스 기반 하이라이트 적용
      let backgroundColor = 'transparent';
      
      for (const [type, indexSet] of Object.entries(highlightMap)) {
        if (indexSet.has(idx)) {
          backgroundColor = HIGHLIGHT_COLORS[type] || 'transparent';
          console.log(`하이라이트 적용: ${type} - 인덱스 ${idx}`);
          break;
        }
      }

      renderedElements.push(
        <Box
          component="span"
          key={idx}
          sx={{
            backgroundColor,
            transition: 'background-color 0.3s',
            display: 'inline',
            padding: '2px 0',
            margin: '1px 0',
            borderRadius: '3px',
          }}
        >
          {sentence}
        </Box>
      );
      
      restOfText = restOfText.substring(originalIndex + sentence.length);
    });

    if (restOfText) {
      renderedElements.push(restOfText);
    }

    return renderedElements;
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', py: 10 }}>
        <CircularProgress />
        <Typography sx={{ ml: 2 }}>분석 결과를 불러오는 중입니다...</Typography>
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
        분석 데이터를 불러올 수 없습니다.
      </Alert>
    );
  }

  const analysis = scriptData.script_analysis;

  const analysisItems = [
    { type: 'uncertainty', label: '추측 표현', count: analysis.uncertainty_count, description: '단정적이지 않고 추측하는 듯한 뉘앙스를 주는 표현입니다.' },
    { type: 'non_honorific', label: '비격식 종결 어미', count: analysis.non_honorific_count, description: '공식적인 발표에 어울리지 않는 비격식적인 문장 종결 형식입니다.' },
    { type: 'subject_verb', label: '주어-서술어 호응', count: analysis.subject_verb_mismatch_count, description: '문장의 주어와 서술어의 관계가 문법적으로 자연스럽지 않은 경우입니다.' },
    { type: 'profanity', label: '비속어', count: analysis.profanity_count, description: '발표의 신뢰도를 떨어뜨릴 수 있는 비속어나 부적절한 단어입니다.' },
  ];

  const problemItems = analysisItems.filter(item => item.count > 0);
  
  const cardHoverStyle = (cardId: string) => ({
    boxShadow: hoveredCard === cardId
      ? '0 14px 28px rgba(0,0,0,0.15), 0 10px 10px rgba(0,0,0,0.12)'
      : '0 4px 8px rgba(0,0,0,0.08)',
    transform: hoveredCard === cardId ? 'translateY(-4px)' : 'none',
    transition: 'all 0.3s ease-in-out',
    position: 'relative',
    overflow: 'hidden',
    '&::after': {
      content: '""',
      position: 'absolute',
      top: 0,
      left: 0,
      width: '100%',
      height: '4px',
      background: '#000',
      opacity: hoveredCard === cardId ? 1 : 0,
      transition: 'opacity 0.3s ease'
    }
  });

  return (
    <Grid container spacing={3}>
      {/* Left Column */}
      <Grid item xs={12} md={5}>
        <Card
          sx={{
            mb: 2,
            borderRadius: 2,
            ...cardHoverStyle('summary')
          }}
          onMouseEnter={() => setHoveredCard('summary')}
          onMouseLeave={() => setHoveredCard(null)}
        >
          <CardContent sx={{ p: 3 }}>
            <Typography variant="h6" fontWeight="bold" gutterBottom>
              대본 요약 및 분량 분석
            </Typography>
            <Divider sx={{ my: 2 }} />
    <Grid container spacing={2}>
              <Grid item xs={6}>
        <Typography variant="body2" color="text.secondary">파일명</Typography>
                  <Typography variant="subtitle1" fontWeight="medium">{scriptData.filename}</Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">발표 시간</Typography>
                  <Typography variant="subtitle1" fontWeight="medium">{scriptData.speech_minutes}분</Typography>
          </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">총 글자수</Typography>
                  <Typography variant="subtitle1" fontWeight="medium">{analysis.length}자</Typography>
          </Grid>
                <Grid item xs={6}>
                  <Typography variant="body2" color="text.secondary">분석일</Typography>
                  <Typography variant="subtitle1" fontWeight="medium">
          {new Date(scriptData.timestamp).toLocaleDateString('ko-KR')}
        </Typography>
      </Grid>
      </Grid>
            <Box sx={{ mt: 3, mb: 1 }}>
              <Alert
                severity="info"
                icon={false}
                sx={{
                  background: 'linear-gradient(90deg, #e3f2fd 0%, #fffde7 100%)',
                  color: '#222',
                  fontWeight: 'bold',
                  fontSize: '1.1rem',
                  borderRadius: 2,
                  px: 2,
                  py: 1.5,
                  mb: 1,
                }}
              >
                권장 글자수: <span style={{ color: '#1976d2', fontWeight: 700 }}>{analysis.min_length}자 ~ {analysis.max_length}자</span>
              </Alert>
      <Alert 
                severity={analysis.length >= analysis.min_length && analysis.length <= analysis.max_length ? 'success' : 'warning'}
                icon={false}
                sx={{ mt: 1 }}
      >
        {analysis.length_feedback}
      </Alert>
            </Box>
          </CardContent>
        </Card>

        <Divider sx={{ my: 2 }} />

        <Card
           sx={{
            borderRadius: 2,
            ...cardHoverStyle('analysis')
          }}
          onMouseEnter={() => setHoveredCard('analysis')}
          onMouseLeave={() => setHoveredCard(null)}
        >
          <CardContent sx={{ p: 3 }}>
            <Typography variant="h6" fontWeight="bold" gutterBottom>
              문장 유형별 분석
              </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
              각 항목을 클릭하면 우측 대본에서 해당 문장이 하이라이트됩니다.
              </Typography>
            <Divider sx={{ mb: 1 }} />
            {problemItems.length > 0 ? (
              <List component="nav" dense>
                {problemItems.map(item => (
                  <ListItem
                    key={item.type}
                    button
                    onClick={() => handleAnalysisTypeClick(item.type)}
                    sx={{
                      my: 1,
                      borderRadius: 2,
                      backgroundColor: activeHighlightTypes.has(item.type) ? HIGHLIGHT_COLORS[item.type] : 'transparent',
                      transition: 'background-color 0.3s, box-shadow 0.3s, transform 0.3s',
                      boxShadow: activeHighlightTypes.has(item.type) ? 3 : 1,
                      '&:hover': {
                        boxShadow: 6,
                        transform: 'translateY(-2px)',
                        backgroundColor: activeHighlightTypes.has(item.type) ? HIGHLIGHT_COLORS[item.type] : 'action.hover',
                      },
                    }}
                  >
                    {ICONS[item.type]}
                    <ListItemText 
                      primary={item.label}
                      primaryTypographyProps={{ fontWeight: activeHighlightTypes.has(item.type) ? 'bold' : 'normal' }}
                      secondary={activeHighlightTypes.has(item.type) ? item.description : null}
                    />
                    <Badge badgeContent={item.count} color="error" />
                  </ListItem>
                ))}
              </List>
            ) : (
              <Box sx={{ textAlign: 'center', py: 4 }}>
                <Alert
                  severity="success"
                  icon={<CheckCircleOutlineIcon />}
                  sx={{
                    background: 'linear-gradient(135deg, #e8f5e8 0%, #f1f8e9 100%)',
                    color: '#2e7d32',
                    fontWeight: 'bold',
                    fontSize: '1.1rem',
                    borderRadius: 3,
                    px: 3,
                    py: 2,
                    border: '2px solid #4caf50',
                    boxShadow: '0 4px 12px rgba(76, 175, 80, 0.2)',
                    '& .MuiAlert-icon': {
                      color: '#4caf50',
                      fontSize: '2rem'
                    }
                  }}
                >
                  모두 정상입니다! 🎉
                </Alert>
                <Typography 
                  variant="body2" 
                  color="text.secondary" 
                  sx={{ mt: 2, fontStyle: 'italic' }}
                >
                  대본에 검토가 필요한 문제점이 발견되지 않았습니다.
                </Typography>
              </Box>
                )}
            </CardContent>
          </Card>
        </Grid>

      {/* Right Column */}
      <Grid item xs={12} md={7}>
        <Card
          sx={{
            height: { md: 'calc(100vh - 200px)' },
            display: 'flex',
            flexDirection: 'column',
            borderRadius: 2,
            ...cardHoverStyle('script')
          }}
          onMouseEnter={() => setHoveredCard('script')}
          onMouseLeave={() => setHoveredCard(null)}
        >
          <CardContent sx={{ flexGrow: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', p: { xs: 2, md: 3 } }}>
            <Box>
              <Typography variant="h6" fontWeight="bold" gutterBottom>
                대본 전문
              </Typography>
              <Divider sx={{ mb: 2 }} />
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mb: 2, alignItems: 'center' }}>
                <Typography variant="subtitle1" sx={{ mr: 1 }}>범례:</Typography>
                {problemItems.map(item => (
                    <Box key={item.type} sx={{ display: 'flex', alignItems: 'center' }}>
                        <Box sx={{ width: 12, height: 12, borderRadius: '50%', backgroundColor: HIGHLIGHT_COLORS[item.type], mr: 0.5, border: '1px solid #ddd' }} />
                        <Typography variant="caption">{item.label}</Typography>
                    </Box>
                ))}
              </Box>
            </Box>
            <Paper
              variant="outlined"
              sx={{
                p: 2.5,
                whiteSpace: 'pre-wrap',
                wordBreak: 'break-all',
                lineHeight: 1.9,
                flexGrow: 1,
                overflowY: 'auto',
                backgroundColor: '#fff',
                fontSize: '1rem',
                borderRadius: 2,
                '::-webkit-scrollbar': {
                  width: '8px',
                  background: '#f5f5f5',
                  borderRadius: '4px',
                },
                '::-webkit-scrollbar-thumb': {
                  background: '#e0e0e0',
                  borderRadius: '4px',
                },
              }}
            >
              {renderHighlightedScript()}
            </Paper>
            </CardContent>
          </Card>
        </Grid>
    </Grid>
  );
};

export default ScriptPage;