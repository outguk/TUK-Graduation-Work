import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import { viteStaticCopy } from 'vite-plugin-static-copy'  // 자동 복사 플러그인 임포트

export default defineConfig({
  plugins: [
    react(),
    viteStaticCopy({ // 여기서 자동 복사 설정 추가
      targets: [
        {
          src: 'dist/**/*', // 빌드 결과물 전체를 복사 대상
          dest: '../../GW-backend/src/main/resources/static' // Spring Boot의 정적 리소스 폴더 경로
        }
      ]
    })
  ],
  server: {
    proxy: {
      '/upload': 'http://localhost:8080',
      '/users': 'http://localhost:8080',
      // 필요하면 더 추가
    }
  },
  build: {           // 선택사항: 빌드 결과물을 관리하기 위한 옵션
    outDir: 'dist', 
    emptyOutDir: true,
  },
});
