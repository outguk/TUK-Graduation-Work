import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { useState } from 'react';

export default function ScriptUpload() {
  const { filename } = useParams<{ filename: string }>();
  const [file, setFile] = useState<File | null>(null);
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) {
      alert('텍스트 파일을 선택해주세요');
      return;
    }

    // ✅ 토큰 읽어서 변수에 저장
    const token = localStorage.getItem('token');   // 또는 context/state

    const formData = new FormData();
    formData.append('file', file);         // FastAPI가 요구하는 필드명은 'file'
    formData.append('filename', filename!);

    await axios.post('/spring/api/script-upload', formData, {
      headers: {
        Authorization: `Bearer ${token ?? ''}`,    // token이 null일 수도 있으니 방어
        'Content-Type': 'multipart/form-data',
      },
    });

    // 분석 완료 후 결과 페이지로 이동
    navigate(`/analysis/${filename}/script-display`);
  };

  return (
    <form onSubmit={handleSubmit}>
      <h2>대본 파일 업로드</h2>
      <input
        type="file"
        accept=".txt"
        onChange={(e) => setFile(e.target.files?.[0] ?? null)}
      />
      <button type="submit">분석 시작</button>
    </form>
  );
}
