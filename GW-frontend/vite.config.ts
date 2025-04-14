import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react-swc'
import { viteStaticCopy } from 'vite-plugin-static-copy'

export default defineConfig({
  plugins: [
    react(),
    // build 후에 실행되도록 설정하고 복사 대상을 세분화
    viteStaticCopy({
      targets: [
        {
          src: 'dist/assets/*', // assets 디렉토리만 복사 대상으로 지정
          dest: '../../GW-backend/src/main/resources/static/assets' // assets 폴더로 직접 복사
        },
        {
          src: 'dist/index.html', // index.html 파일 복사
          dest: '../../GW-backend/src/main/resources/static' // 루트에 복사
        }
      ],
      hook: 'writeBundle' // 빌드 완료 후 실행되도록 설정
    })
  ],
  server: {
    proxy: {
      '/upload': 'http://localhost:8080',
      '/users': 'http://localhost:8080',
      '/spring/api': 'http://localhost:8080',
      '/fastapi/api': 'http://localhost:5000',
    }
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    // assets 파일은 assets/ 폴더에만 생성되도록 설정
    assetsDir: 'assets',
    // assets 내부 경로 형식 지정
    rollupOptions: {
      output: {
        assetFileNames: 'assets/[name].[hash].[ext]',
        chunkFileNames: 'assets/[name].[hash].js',
        entryFileNames: 'assets/[name].[hash].js'
      }
    }
  }
});