# GW-Frontend 졸업작품 React 협업 가이드
---

## 📦 프로젝트 개요

=======
React + Vite + TypeScript + MUI 기반으로 구성된 프론트엔드. 백엔드는 Spring Boot와 연결되어 있으며, FastAPI 분석 서버도 함께 통신

---

## 🛠️ 개발 환경

- Node.js = 20.19.0   
- npm  = 10.8.2

## 버전 확인
node -v
npm -v
---

## 📥 설치 및 실행(node와 npm은 미리 깔아야 함)

### 1. 프로젝트 클론
```bash
git clone <레포 주소>
cd GW-frontend
```

### 2. 패키지 설치
```bash
npm install # 올려둔 pakege.json 을 토대로 하위 패키지들이 설치됨
npm install @mui/material @emotion/react @emotion/styled # MUI 디자인 프레임워크 설치
npm install react-router-dom # 프론트엔드에서 페이지 전환을 위해서
```
=======
---

## 🔄 Vite 프록시 설정 (`vite.config.ts`) 및 정적 리소스 자동 복사 설정(이미 되어있음)
- vite.config.ts 파일에서 프록시 서버를 생성에 spring과 연동되도록 설정

```ts
server: {
  proxy: {
    '/upload': 'http://localhost:8080',
    '/users': 'http://localhost:8080'
  }
}

viteStaticCopy({
  targets: [
    { src: 'dist/**/*', dest: '../../GW-backend/src/main/resources/static' }
  ]
})
```

## ⚙️ 빌드

### 빌드(GW-frontend 디렉토리에서 실행)
```bash
npm run build
```
---
- 빌드 시 Spring Boot의 static에 React 정적 리소스 파일인 dist 밑의 파일들이 자동 복사됨(Spring과 연결)

## 빌드 시 주의 사항
=======
- frontend 코드를 변경하면 빌드를 다시 해야 연동되는데 이때 Spring의 static에 js파일이 덮어씌어지는 것이 아닌 하나 더 생김. 이전 js파일을 지워야 함
---

### 3. 백엔드 연동
- Spring Boot 백엔드(`localhost:8080`)와 AI서버를 모두 실행해야 업로드/회원가입 API 작동함

## 📂 Git 업로드 규칙

### ✅ 반드시 포함할 파일
- `src/`, `public/`
- `vite.config.ts`, `tsconfig.json`
- `package.json`, `package-lock.json`

### ❌ 포함하지 말아야 할 파일
- `node_modules/`
- `dist/`
- `.env`

> `.gitignore`에 설정되어 있어야 함


+ Controller 부분에 변경점이 많아서 주석으로 변경점 작성해놓음

  ### 4/3 업데이트 내용
  - 발성 분석 코드 최적화 -> 정확한 비동기 및 병렬 처리로 효율 및 속도 상승(비언어 주석 처리 되어있음)
  - 기존 로그인 페이지였던 HomPage.tsx를 서비스 소개 페이지로 변경하고, 로그인 페이지를 SignInPage로 따로 만듬. 홈페이지의 분석 시작 버튼을 누르면 로그인 페이지로 이동
    
  **- 현재 자동 설치된 react 버전이 19.x 버전에 MUI 버전이 7.X 버전이었는데 안정성 문제로 디자인에서 오류가 자주 발생해서 react와 MUI 버전을 다운그레이드 함**
  - ![image](https://github.com/user-attachments/assets/387dc1ba-3d48-43f5-859c-029cce229692)
    
  **- 위 이미지처럼 버전을 변경해주어야 홈페이지 디자인이 잘 적용됨**



