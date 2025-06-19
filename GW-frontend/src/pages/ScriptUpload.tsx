import { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

export default function ScriptUpload() {
  const [file, setFile] = useState<File | null>(null);
  const [minutes, setMinutes] = useState<number>(1); // 기본값 1분
  const navigate = useNavigate();

  // 파일 업로드 제출 핸들러
  const handleFileSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return alert('텍스트 파일을 선택해주세요');
    if (minutes < 1) return alert('발표 시간을 1분 이상 입력하세요');

    const token = localStorage.getItem('token');
    if (!token) return alert('로그인이 필요합니다');

    const formData = new FormData();
    formData.append('scriptFile', file);
    formData.append('filename', file.name);
    formData.append('speech_minutes', minutes.toString());

    try {
      const response = await axios.post('/spring/api/script-upload', formData, {
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'multipart/form-data',
        },
      });
      if (response.data && response.data._id) {
        console.log('분석 성공, script_id:', response.data._id);
        // AnalysisDashboardPage로 리디렉션하여 대본 분석 결과 표시
        navigate(`/analysis/${response.data._id}`);
      } else {
        console.error('응답 데이터 구조:', response.data);
        alert('분석 실패: 응답에서 script_id를 찾을 수 없습니다');
      }
    } catch (error) {
      alert('대본 분석 실패: ' + (error as any).message);
    }
  };

  return (
    <div style={{ padding: '20px' }}>
      <h2>대본 분석</h2>
      <form onSubmit={handleFileSubmit}>
        <label>
          발표 시간(분):
          <input
            type="number"
            min={1}
            value={minutes}
            onChange={(e) => setMinutes(Number(e.target.value))}
            style={{ marginLeft: '8px', marginBottom: '20px' }}
          />
        </label>
        <h3>대본 파일 업로드</h3>
        <input
          type="file"
          accept=".txt"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        <button
          type="submit"
          disabled={!file || minutes < 1}
          style={{ display: 'block', marginTop: '12px' }}
        >
          파일 분석 시작
        </button>
      </form>
    </div>
  );
}