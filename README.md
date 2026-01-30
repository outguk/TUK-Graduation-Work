
<h1 align="center">$\bf{\large{\color{#6580DD} AI \ 기반 \ 발표 \ 분석 \ 및 \ 피드백 \ 시스템 }}$</h1>

<p align="center">
영상·음성·대본을 종합 분석하여 발표 역량 향상을 돕는 멀티모달 AI 플랫폼
</p>

---

## 개발 환경

### Language
![Java](https://img.shields.io/badge/java-%23ED8B00.svg?style=for-the-badge&logo=openjdk&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TypeScript](https://img.shields.io/badge/typescript-%23007ACC.svg?style=for-the-badge&logo=typescript&logoColor=white)

### Framework & Runtime
![Spring](https://img.shields.io/badge/spring-%236DB33F.svg?style=for-the-badge&logo=spring&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![React](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)
![Vite](https://img.shields.io/badge/vite-%23646CFF.svg?style=for-the-badge&logo=vite&logoColor=white)

### Database
![MySQL](https://img.shields.io/badge/mysql-4479A1.svg?style=for-the-badge&logo=mysql&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-%234ea94b.svg?style=for-the-badge&logo=mongodb&logoColor=white)

### AI / ML
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![OpenAI](https://img.shields.io/badge/Whisper-412991?style=for-the-badge&logo=openai&logoColor=white)

### Infra
![AWS](https://img.shields.io/badge/AWS_RDS-%23FF9900.svg?style=for-the-badge&logo=amazonaws&logoColor=white)
![MongoDB Atlas](https://img.shields.io/badge/MongoDB_Atlas-%234ea94b.svg?style=for-the-badge&logo=mongodb&logoColor=white)

<hr>

## Key Dependencies and Features

### 1. 멀티모달 발표 분석 파이프라인
- 발표 영상을 업로드하면 **음성 분석**, **비언어 분석**, **대본 분석**을 자동 수행
- 7단계 비동기 파이프라인으로 구성되어 실시간 진행률 추적 가능
- UUID 기반 태스크 관리로 동시 다수 분석 요청 처리

### 2. 음성 분석 (Vocalization Analysis)
- OpenAI Whisper 기반 음성-텍스트 변환 (STT)
- 발화 속도(WPM) 구간별 측정 및 평가
- 음량·음성 크기 분석을 통한 전달력 평가

### 3. 비언어 분석 (Non-Verbal Analysis)
- MMPose 기반 골격(Skeleton) 키포인트 추출
- ST-GCN++ 그래프 합성곱 네트워크를 활용한 제스처·동작 인식
- 프레임 단위 자세 분석 및 행동 분류

### 4. 대본 분석 (Script Analysis)
- KoNLPy·KorCen 기반 한국어 자연어 처리
- 경어법 일관성, 격식체 사용 여부 검증
- 불확실 표현 탐지 및 비속어 필터링
- 문법·구조 평가를 통한 대본 품질 점수 산출

### 5. 비동기 리액티브 아키텍처
- Spring WebFlux 기반 논블로킹 I/O 처리
- FastAPI + asyncio 기반 AI 분석 비동기 수행
- ThreadPoolExecutor를 활용한 CPU 바운드 작업 병렬 처리

### 6. JWT 기반 인증·인가
- Spring Security + JWT 토큰 기반 사용자 인증
- BCrypt 패스워드 해싱
- 보호된 API 엔드포인트에 대한 토큰 검증

<hr>

## 아키텍처

### 시스템 아키텍처

본 시스템은 **프론트엔드 → Spring Boot 백엔드 → FastAPI AI 서버**의 3-Tier 구조로 설계되어 있습니다. <br>
Spring Boot 백엔드는 사용자 인증·인가 및 API 프록시 역할을 수행하며, <br>
FastAPI AI 서버는 영상·음성·대본에 대한 딥러닝 기반 분석을 담당합니다. <br>
분석 결과는 MongoDB에, 사용자 정보는 AWS RDS MySQL에 분리 저장하여 데이터 독립성을 보장합니다.

<br>

| 서비스 | 기술 스택 | 포트 | 설명 |
| --- | --- | --- | --- |
| **Frontend** | React 19 + TypeScript + Vite | 5173 | 사용자 인터페이스, 분석 대시보드, 차트 시각화 |
| **Backend** | Spring Boot 3.4 + WebFlux | 8080 | 인증·인가, API 라우팅, 정적 리소스 서빙 |
| **AI Server** | FastAPI + PyTorch | 5000 | 음성·비언어·대본 분석 엔진 |
| **RDB** | AWS RDS MySQL | - | 사용자 계정 및 프로필 관리 |
| **NoSQL** | MongoDB Atlas | - | 분석 결과 및 대본 피드백 저장 |

<br>

<p align="center">
  <img src="./images/시스템 아키텍처.png" alt="시스템 아키텍처" width="800"/>
</p>

<br>

### 분석 파이프라인 흐름

```
영상 업로드 → 오디오 추출 → 전처리(리샘플링, 노이즈 제거)
    → Whisper STT 변환 → 발화 속도·음량 분석
    → 프레임 추출 → 키포인트 검출 → ST-GCN++ 동작 인식
    → 결과 통합 → MongoDB 저장 → 대시보드 시각화
```

<hr>

## Component & API URI Collection

### Upload & Analysis Component
영상 업로드 및 AI 분석을 처리하는 컴포넌트

| URI | Method | 설명 |
| --- | --- | --- |
| `/spring/api/upload` | POST | MP4 영상 업로드 및 분석 요청 |
| `/spring/api/my-analyses` | GET | 사용자의 분석 결과 목록 조회 |
| `/spring/api/get-analysis?filename=X` | GET | 특정 분석 결과 상세 조회 |
| `/spring/api/video/{filename}` | GET | 분석 영상 스트리밍 |
| `/fastapi/api/upload-video/` | POST | 영상 분석 파이프라인 실행 |
| `/fastapi/api/progress/{task_id}` | GET | 분석 진행률 실시간 조회 |

<br>

### Script Analysis Component
대본 분석을 담당하는 컴포넌트

| URI | Method | 설명 |
| --- | --- | --- |
| `/spring/api/script-upload` | POST | TXT 대본 파일 업로드 및 분석 |
| `/spring/api/analyze-text-script` | POST | 텍스트 입력 대본 분석 |
| `/spring/api/get-script-analysis?script_id=X` | GET | 대본 분석 결과 조회 |
| `/fastapi/api/analyze-script/` | POST | 대본 파일 분석 수행 |
| `/fastapi/api/analyze-text-script/` | POST | 텍스트 대본 분석 수행 |

<br>

### User Component
사용자 인증 및 프로필 관리를 담당하는 컴포넌트

| URI | Method | 설명 |
| --- | --- | --- |
| `/spring/api/users/new` | POST | 회원가입 |
| `/spring/api/users/login` | POST | 로그인 (JWT 토큰 발급) |
| `/spring/api/user/profile` | GET | 사용자 프로필 조회 |
| `/spring/api/user/profile` | PUT | 사용자 정보 수정 |

<hr>

## 프론트엔드 페이지 구성

| 페이지 | 설명 |
| --- | --- |
| **HomePage** | 서비스 소개 랜딩 페이지 |
| **SignIn / SignUp** | 로그인 및 회원가입 |
| **MainPage** | 로그인 후 메인 대시보드 |
| **UploadPage** | 영상 업로드 및 실시간 진행률 표시 |
| **AnalysisDashboardPage** | 분석 결과 종합 대시보드 (Recharts 차트) |
| **SpeechEvaluationDetailPage** | 발화 속도·음량 상세 분석 그래프 |
| **NonverbalEvaluationDetailPage** | 제스처·자세 분석 상세 결과 |
| **ScriptUpload** | 대본 업로드 및 텍스트 입력 분석 |
| **UserProfilePage** | 사용자 프로필 관리 |

<br>

### 페이지 스크린샷

<p align="center">
  <img src="./images/초기 메인화면.png" alt="초기 메인화면" width="700"/>
  <br><em>초기 메인화면</em>
</p>

<br>

<p align="center">
  <img src="./images/로그인.png" alt="로그인" width="700"/>
  <br><em>로그인</em>
</p>

<br>

<p align="center">
  <img src="./images/회원가입.png" alt="회원가입" width="700"/>
  <br><em>회원가입</em>
</p>

<br>

<p align="center">
  <img src="./images/발표영상업로드 페이지.png" alt="발표영상 업로드 페이지" width="700"/>
  <br><em>발표영상 업로드 페이지</em>
</p>

<br>

<p align="center">
  <img src="./images/전체대시보드.png" alt="전체 대시보드" width="700"/>
  <br><em>분석 결과 종합 대시보드</em>
</p>

<br>

<p align="center">
  <img src="./images/말하기속도 분석 대시보드.png" alt="말하기속도 분석 대시보드" width="700"/>
  <br><em>말하기 속도 분석 대시보드</em>
</p>

<br>

<p align="center">
  <img src="./images/음량 분석 대시보드.png" alt="음량 분석 대시보드" width="700"/>
  <br><em>음량 분석 대시보드</em>
</p>

<br>

<p align="center">
  <img src="./images/비언어분석 대시보드.png" alt="비언어 분석 대시보드" width="700"/>
  <br><em>비언어 분석 대시보드</em>
</p>

<br>

<p align="center">
  <img src="./images/대본 분석 대시보드.png" alt="대본 분석 대시보드" width="700"/>
  <br><em>대본 분석 대시보드</em>
</p>

<hr>

## Database Schema

### MySQL (AWS RDS) - 사용자 정보

| 필드 | 타입 | 설명 |
| --- | --- | --- |
| `id` | INT (PK) | 자동 증가 식별자 |
| `username` | VARCHAR (UNIQUE) | 사용자명 |
| `password` | VARCHAR | BCrypt 해싱된 비밀번호 |
| `email` | VARCHAR (UNIQUE) | 이메일 주소 |
| `created_at` | TIMESTAMP | 계정 생성 시각 |

### MongoDB Atlas - 분석 결과

**results 컬렉션**
```json
{
  "user_id": 1,
  "filename": "presentation.mp4",
  "speaking_speed": { "wpm": 135, "segments": [...] },
  "speaking_evaluation": 82.5,
  "volume_analysis": { "avg_db": -20.3, "segments": [...] },
  "volume_evaluation": 75.0,
  "nonverbal_analysis": [ { "action": "gesture", "confidence": 0.92 } ],
  "timestamp": "2025-01-15T10:30:00Z"
}
```

**script_analysis 컬렉션**
```json
{
  "user_id": 1,
  "script_text": "발표 대본 내용...",
  "script_analysis": { "formality_score": 88, "feedback": [...] },
  "speech_minutes": 15,
  "filename": "script.txt",
  "timestamp": "2025-01-15T11:00:00Z"
}
```

<hr>

## 실행 방법

### 1. Backend (Spring Boot)
```bash
cd GW-backend
./gradlew build
java -jar build/libs/GW-backend-0.0.1-SNAPSHOT.jar
# → localhost:8080
```

### 2. AI Server (FastAPI)
```bash
cd python-AI
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 5000
# → localhost:5000
```

### 3. Frontend (React + Vite)
```bash
cd GW-frontend
npm install
npm run dev
# → localhost:5173
```

> **Note**: 프론트엔드 빌드 시 `npm run build`를 실행하면 빌드 결과물이 자동으로 `GW-backend/src/main/resources/static/`에 복사되어 Spring Boot에서 정적 리소스로 서빙됩니다.

