// src/styles/theme.ts
import { createTheme } from '@mui/material/styles';

// 한국어 폰트
const theme = createTheme({
  typography: {
    fontFamily: [
      'Pretendard',
      '-apple-system',
      'BlinkMacSystemFont',
      'sans-serif'
    ].join(','),
    h1: {
      fontWeight: 700,
      letterSpacing: '-0.01em',
    },
    h2: {
      fontWeight: 600,
      letterSpacing: '-0.01em',
    },
    body1: {
      fontWeight: 400,
    }
  },
  components: {
    MuiCssBaseline: {
      styleOverrides: `
        @keyframes pulse {
          0%, 100% {
            opacity: 0.7;
          }
          50% {
            opacity: 0.4;
          }
        }
      `,
    },
  },
});

export default theme;
