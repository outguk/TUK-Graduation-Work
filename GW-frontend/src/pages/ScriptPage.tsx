import { useParams } from 'react-router-dom'
import { useEffect, useState } from 'react'
import axios from 'axios'

interface ScriptAnalysis {
  length: number
  length_feedback: string
  uncertainty_examples: [number, string][]
  // … JSON 필드 그대로 타입 정의
}

export default function ScriptPage() {
  const { filename } = useParams<{ filename: string }>()
  const [analysis, setAnalysis] = useState<ScriptAnalysis | null>(null)

  useEffect(() => {
    const token = localStorage.getItem('token');
    axios.get('/spring/api/get-analysis', {
      params: { filename },
      headers: { 'Authorization': `Bearer ${token}` }
    }).then(res => {
      setAnalysis(res.data.script_analysis)
    })
  }, [filename])

  if (!analysis) return <div>로딩 중…</div>
  return (
    <div>
      <h2>대본 분석 결과</h2>
      <p>총 글자수: {analysis.length}</p>
      <p>피드백: {analysis.length_feedback}</p>
      <h3>불확실 표현 예시</h3>
      <ul>
        {analysis.uncertainty_examples.map(([pos, text], i) =>
          <li key={i}>{pos}번째 단어: “{text}”</li>
        )}
      </ul>
      {/* 나머지 피드백 항목도 유사하게 렌더링 */}
    </div>
  )
}
