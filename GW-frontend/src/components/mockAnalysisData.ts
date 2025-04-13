/**
 * mockAnalysisData.ts
 * 
 * 백엔드 개발자를 위한 안내:
 * 이 파일은 프론트엔드 개발 중 백엔드 API 연동 전에 사용하는 임시 데이터입니다.
 * 실제 구현 시 이 파일의 데이터 구조를 참고하여 API 응답을 설계해야 합니다.
 * 
 * 모든 타입 정의와 데이터 형식은 백엔드에서 제공해야 할 API 응답 구조를 나타냅니다.
 */

/**
 * 사이드바에 표시되는 발표 목록 항목 타입
 * 
 * 백엔드 개발 참고사항:
 * - GET /api/presentations 엔드포인트의 응답 형식으로 사용됩니다.
 * - 모든 필드는 필수입니다.
 */

// 사이드바에 표시되는 발표 정보 타입
export interface PresentationEntry {
  id: string; // 발표 고유 식별자 (UUID 또는 기타 고유 문자열)
  title: string; // 발표 제목
  date: string; // 발표 날짜 (YYYY.MM.DD 형식 권장)
  duration: string;   // 발표 길이 (M:SS 형식, 예: "0:45")
}

// 말하기 속도 세그먼트 타입
export interface SpeakingSpeedSegment {
  start: number; // 구간 시작 시간 (초 단위)
  end: number; // 구간 종료 시간 (초 단위)
  wpm: number; // 분당 단어 수 (Words Per Minute)
  words: string; // 해당 구간에서 말한 텍스트
}

// 말하기 속도 데이터 타입
export interface SpeakingSpeedData {
  overall_wpm: number; // 전체 발표의 평균 WPM
  total_spoken_time: number; // 총 발화 시간 (초 단위)
  segment_wpm: SpeakingSpeedSegment[];// 구간별 WPM 데이터
}

// 말하기 평가 세그먼트 타입
export interface SpeakingEvaluationSegment {
  start: number; // 구간 시작 시간 (초 단위)
  end: number;  // 구간 종료 시간 (초 단위)
  wpm: number;  // 분당 단어 수 (Words Per Minute)
  feedback: string;  // 피드백 메시지 (예: "적절한 속도입니다", "약간 빠릅니다")
  score: number; // 평가 점수 (0-100)
}

// 말하기 평가 데이터 타입
export interface SpeakingEvaluationData {
  segment_evaluations: SpeakingEvaluationSegment[]; // 구간별 평가 데이터
  overall_score: number;  // 전체 평가 점수 (0-100)
}

// 음량 분석 세그먼트 타입
export interface VolumeSegmentData {
  time_stamps: number[]; // 구간의 시작/종료 시간 (초 단위의 배열, [start, end])
  rms: number; // RMS 음량값
  db: number; // 데시벨(dB) 값
}

// 음량 분석 데이터 타입
export interface VolumeAnalysisData {
  segment_data: VolumeSegmentData[]; // 구간별 음량 데이터
  mean_rms: number;                 // 평균 RMS 값
  mean_db: number;                  // 평균 데시벨(dB) 값
}

// 음량 평가 세그먼트 타입
export interface VolumeEvaluationSegment {
  time_stamps: number[];  // 구간의 시작/종료 시간 (초 단위의 배열, [start, end])
  db: number;            // 데시벨(dB) 값
  feedback: string;      // 피드백 메시지 (예: "적절한 음량입니다", "약간 조용함")
  score: number;         // 평가 점수 (0-100)
}

// 음량 평가 데이터 타입
export interface VolumeEvaluationData {
  segment_evaluations: VolumeEvaluationSegment[]; // 구간별 평가 데이터
  overall_score: number;                        // 전체 평가 점수 (0-100)
}

// 행동 클래스 타입
export interface BehaviorClass {
  class: string;        // 행동 유형 (예: "자세(비스듬히)", "손동작(얼굴)")
  probability: number;  // 확률 (0-100)
}

// 비언어적 분석 세그먼트 타입
export interface NonverbalSegment {
  sample_number: number;     // 샘플 번호 (순차적)
  time_range: string;        // 시간 범위 (문자열, 예: "0.00s ~ 10.00s")
  frame_range: string;       // 프레임 범위 (문자열, 예: "0 ~ 49")
  frame_dir: string;         // 프레임 디렉토리 (예: "frame_0")
  is_normal: boolean;        // 정상 행동 여부
  wrist_distance?: number;   // 손목 간 거리 (선택적)
  top_classes?: BehaviorClass[]; // 감지된 행동 클래스 (확률 내림차순)
  is_hands_behind_detected?: boolean; // 손이 뒤로 있는지 여부 (선택적)
}


/**
 * 발표 분석 결과 전체 타입
 * 
 * 백엔드 개발 참고사항:
 * - 발표의 모든 분석 결과를 포함하는 주요 데이터 구조입니다.
 * - GET /api/presentations/:id 엔드포인트의 응답 형식으로 사용됩니다.
 */
export interface PresentationAnalysis {
  id: string;                        // 발표 고유 식별자
  title: string;                     // 발표 제목
  date: string;                      // 발표 날짜 (YYYY.MM.DD 형식)
  duration: string;                  // 발표 길이 (M:SS 형식)
  user_id: number;                   // 사용자 ID
  filename: string;                  // 원본 영상 파일명
  speaking_speed: SpeakingSpeedData;          // 말하기 속도 분석 결과
  speaking_evaluation: SpeakingEvaluationData; // 말하기 평가 결과
  volume_analysis: VolumeAnalysisData;         // 음량 분석 결과
  volume_evaluation: VolumeEvaluationData;     // 음량 평가 결과
  nonverbal_analysis: NonverbalSegment[];      // 비언어적 분석 결과
}

// 차트 데이터 포인트 타입
export interface ChartDataPoint {
  time: string;
  wpm?: number;
  db?: number;
  timeRange?: string;
}

// 목업 발표 목록 데이터 (사이드바 표시용)
export const mockPresentations: PresentationEntry[] = [
    {
      id: "pres-001",
      title: "팀 프로젝트 발표",
      date: "2025.04.10",
      duration: "0:45",
    },
    {
      id: "pres-002",
      title: "취업 인터뷰 연습",
      date: "2025.04.03",
      duration: "0:52",
    },
    {
      id: "pres-003",
      title: "신제품 소개",
      date: "2025.03.25",
      duration: "0:58",
    }
  ];
  
  // 목업 발표 분석 데이터 - 첫 번째 발표
  export const presentation1: PresentationAnalysis = {
    "id": "pres-001",
    "title": "팀 프로젝트 발표",
    "date": "2025.04.10",
    "duration": "0:45",
    "user_id": 2,
    "filename": "team_project.mp4",
    "speaking_speed": {
      "overall_wpm": 125.63,
      "total_spoken_time": 45.12,
      "segment_wpm": [
        {
          "start": 0.06,
          "end": 15.81,
          "wpm": 126.87,
          "words": "안녕하세요. 오늘 저희 팀에서 준비한 프로젝트 발표를 시작하겠습니다. 이번 프로젝트는 사용자 경험 향상을 위한 인터페이스 개선에 관한 내용입니다."
        },
        {
          "start": 16.42,
          "end": 30.05,
          "wpm": 131.24,
          "words": "먼저 기존 시스템의 문제점을 분석한 결과를 말씀드리겠습니다. 사용자 인터뷰와 데이터 분석을 통해 다음과 같은 문제점을 발견했습니다."
        },
        {
          "start": 31.48,
          "end": 45.12,
          "wpm": 120.12,
          "words": "다음은 저희가 제안하는 해결책입니다. 첫째, 사용자 흐름을 단순화했습니다. 둘째, 직관적인 네비게이션 시스템을 도입했습니다."
        }
      ]
    },
    "speaking_evaluation": {
      "segment_evaluations": [
        {
          "start": 0.06,
          "end": 15.81,
          "wpm": 126.87,
          "feedback": "적절한 속도입니다.",
          "score": 96.87
        },
        {
          "start": 16.42,
          "end": 30.05,
          "wpm": 131.24,
          "feedback": "약간 빠릅니다. (-1.24 WPM 초과)",
          "score": 98.76
        },
        {
          "start": 31.48,
          "end": 45.12,
          "wpm": 120.12,
          "feedback": "적절한 속도입니다.",
          "score": 95.12
        }
      ],
      "overall_score": 96.92
    },
    "volume_analysis": {
      "segment_data": [
        {
          "time_stamps": [
            0.06,
            15.81
          ],
          "rms": 68.42,
          "db": 58.76
        },
        {
          "time_stamps": [
            16.42,
            30.05
          ],
          "rms": 72.14,
          "db": 60.88
        },
        {
          "time_stamps": [
            31.48,
            45.12
          ],
          "rms": 70.36,
          "db": 59.97
        }
      ],
      "mean_rms": 70.31,
      "mean_db": 59.87
    },
    "volume_evaluation": {
      "segment_evaluations": [
        {
          "time_stamps": [
            0.06,
            15.81
          ],
          "db": 58.76,
          "feedback": "약간 조용함 (-1.24 dB 부족)",
          "score": 88.76
        },
        {
          "time_stamps": [
            16.42,
            30.05
          ],
          "db": 60.88,
          "feedback": "적절한 음량입니다.",
          "score": 90.88
        },
        {
          "time_stamps": [
            31.48,
            45.12
          ],
          "db": 59.97,
          "feedback": "적절한 음량입니다.",
          "score": 89.97
        }
      ],
      "overall_score": 89.87
    },
    "nonverbal_analysis": [
      {
        "sample_number": 1,
        "time_range": "0.00s ~ 10.00s",
        "frame_range": "0 ~ 49",
        "frame_dir": "frame_0",
        "is_normal": true
      },
      {
        "sample_number": 2,
        "time_range": "10.00s ~ 20.00s",
        "frame_range": "50 ~ 99",
        "frame_dir": "frame_50",
        "is_normal": false,
        "wrist_distance": 126.19,
        "top_classes": [
          {
            "class": "머리동작(고개흔들기)",
            "probability": 78.42
          },
          {
            "class": "자세(비비꼬기)",
            "probability": 15.73
          },
          {
            "class": "손동작(몸긁기)",
            "probability": 5.85
          }
        ]
      },
      {
        "sample_number": 3,
        "time_range": "20.00s ~ 30.00s",
        "frame_range": "100 ~ 149",
        "frame_dir": "frame_100",
        "is_normal": true
      },
      {
        "sample_number": 4,
        "time_range": "30.00s ~ 40.00s",
        "frame_range": "150 ~ 199",
        "frame_dir": "frame_150",
        "is_normal": false,
        "wrist_distance": 133.73,
        "top_classes": [
          {
            "class": "자세(비스듬히)",
            "probability": 92.31
          },
          {
            "class": "자세(좌우흔들기)",
            "probability": 6.85
          },
          {
            "class": "손동작(얼굴)",
            "probability": 0.84
          }
        ]
      },
      {
        "sample_number": 5,
        "time_range": "40.00s ~ 45.12s",
        "frame_range": "200 ~ 225",
        "frame_dir": "frame_200",
        "is_normal": true
      }
    ]
  };
  
  // 목업 발표 분석 데이터 - 두 번째 발표
  export const presentation2: PresentationAnalysis = {
    "id": "pres-002",
    "title": "취업 인터뷰 연습",
    "date": "2025.04.03",
    "duration": "0:52",
    "user_id": 2,
    "filename": "interview_practice.mp4",
    "speaking_speed": {
      "overall_wpm": 138.27,
      "total_spoken_time": 52.08,
      "segment_wpm": [
        {
          "start": 0.86,
          "end": 18.41,
          "wpm": 142.63,
          "words": "안녕하세요. 저는 디지털 마케팅 분야에서 3년간 경험을 쌓은 지원자 홍길동입니다. 귀사의 마케팅 전략 포지션에 지원하게 되어 기쁘게 생각합니다."
        },
        {
          "start": 19.25,
          "end": 36.82,
          "wpm": 148.92,
          "words": "제가 가진 가장 큰 강점은 데이터 기반 의사결정과 창의적인 캠페인 기획 능력입니다. 이전 회사에서는 소셜 미디어 마케팅을 통해 고객 참여도를 45% 향상시킨 경험이 있습니다."
        },
        {
          "start": 37.63,
          "end": 52.08,
          "wpm": 128.75,
          "words": "최근 디지털 마케팅 트렌드 중 가장 주목하고 있는 것은 개인화된 콘텐츠 마케팅과 AI를 활용한 타겟팅입니다. 이러한 기술을 활용하면 마케팅 효율성을 크게 높일 수 있다고 생각합니다."
        }
      ]
    },
    "speaking_evaluation": {
      "segment_evaluations": [
        {
          "start": 0.86,
          "end": 18.41,
          "wpm": 142.63,
          "feedback": "약간 빠름 (-12.63 WPM 초과)",
          "score": 87.37
        },
        {
          "start": 19.25,
          "end": 36.82,
          "wpm": 148.92,
          "feedback": "빠름 (-18.92 WPM 초과)",
          "score": 81.08
        },
        {
          "start": 37.63,
          "end": 52.08,
          "wpm": 128.75,
          "feedback": "적절한 속도입니다.",
          "score": 98.75
        }
      ],
      "overall_score": 89.07
    },
    "volume_analysis": {
      "segment_data": [
        {
          "time_stamps": [
            0.86,
            18.41
          ],
          "rms": 66.31,
          "db": 57.64
        },
        {
          "time_stamps": [
            19.25,
            36.82
          ],
          "rms": 71.94,
          "db": 60.72
        },
        {
          "time_stamps": [
            37.63,
            52.08
          ],
          "rms": 69.63,
          "db": 59.48
        }
      ],
      "mean_rms": 69.29,
      "mean_db": 59.28
    },
    "volume_evaluation": {
      "segment_evaluations": [
        {
          "time_stamps": [
            0.86,
            18.41
          ],
          "db": 57.64,
          "feedback": "약간 조용함 (-2.36 dB 부족)",
          "score": 87.64
        },
        {
          "time_stamps": [
            19.25,
            36.82
          ],
          "db": 60.72,
          "feedback": "적절한 음량입니다.",
          "score": 90.72
        },
        {
          "time_stamps": [
            37.63,
            52.08
          ],
          "db": 59.48,
          "feedback": "약간 조용함 (-0.52 dB 부족)",
          "score": 89.48
        }
      ],
      "overall_score": 89.28
    },
    "nonverbal_analysis": [
      {
        "sample_number": 1,
        "time_range": "0.00s ~ 10.00s",
        "frame_range": "0 ~ 49",
        "frame_dir": "frame_0",
        "is_normal": false,
        "wrist_distance": 105.23,
        "top_classes": [
          {
            "class": "팔동작(뒷짐)",
            "probability": 95.68
          },
          {
            "class": "손동작(머리)",
            "probability": 3.22
          },
          {
            "class": "손동작(몸긁기)",
            "probability": 1.10
          }
        ],
        "is_hands_behind_detected": true
      },
      {
        "sample_number": 2,
        "time_range": "10.00s ~ 20.00s",
        "frame_range": "50 ~ 99",
        "frame_dir": "frame_50",
        "is_normal": true
      },
      {
        "sample_number": 3,
        "time_range": "20.00s ~ 30.00s",
        "frame_range": "100 ~ 149",
        "frame_dir": "frame_100",
        "is_normal": false,
        "wrist_distance": 127.74,
        "top_classes": [
          {
            "class": "손동작(손톱)",
            "probability": 82.17
          },
          {
            "class": "손동작(몸긁기)",
            "probability": 14.36
          },
          {
            "class": "자세(비스듬히)",
            "probability": 3.47
          }
        ]
      },
      {
        "sample_number": 4,
        "time_range": "30.00s ~ 40.00s",
        "frame_range": "150 ~ 199",
        "frame_dir": "frame_150",
        "is_normal": true
      },
      {
        "sample_number": 5,
        "time_range": "40.00s ~ 50.00s",
        "frame_range": "200 ~ 249",
        "frame_dir": "frame_200",
        "is_normal": false,
        "wrist_distance": 126.19,
        "top_classes": [
          {
            "class": "손동작(얼굴)",
            "probability": 89.32
          },
          {
            "class": "머리동작(고개흔들기)",
            "probability": 6.74
          },
          {
            "class": "자세(비스듬히)",
            "probability": 3.94
          }
        ]
      },
      {
        "sample_number": 6,
        "time_range": "50.00s ~ 52.08s",
        "frame_range": "250 ~ 260",
        "frame_dir": "frame_250",
        "is_normal": true
      }
    ]
  };
  
  // 목업 발표 분석 데이터 - 세 번째 발표
  export const presentation3: PresentationAnalysis = {
    "id": "pres-003",
    "title": "신제품 소개",
    "date": "2025.03.25",
    "duration": "0:58",
    "user_id": 2,
    "filename": "new_product.mp4",
    "speaking_speed": {
      "overall_wpm": 115.82,
      "total_spoken_time": 58.23,
      "segment_wpm": [
        {
          "start": 0.46,
          "end": 20.18,
          "wpm": 112.37,
          "words": "안녕하세요, 오늘 저희가 새롭게 선보이는 제품에 대해 소개해 드리겠습니다. 이 제품은 1년간의 연구 개발 끝에 완성된 혁신적인 솔루션입니다."
        },
        {
          "start": 21.33,
          "end": 40.56,
          "wpm": 110.85,
          "words": "먼저 기존 시장의 문제점에 대해 살펴보겠습니다. 현재 사용자들이 가장 불편해하는 부분은 다음과 같습니다. 첫째, 복잡한 설정 과정. 둘째, 높은 유지 비용입니다."
        },
        {
          "start": 41.87,
          "end": 58.23,
          "wpm": 124.24,
          "words": "저희 제품은 이러한 문제를 해결하기 위해 직관적인 인터페이스와 저비용 유지관리 시스템을 갖추고 있습니다. 이제 실제 제품 데모를 보여드리겠습니다."
        }
      ]
    },
    "speaking_evaluation": {
      "segment_evaluations": [
        {
          "start": 0.46,
          "end": 20.18,
          "wpm": 112.37,
          "feedback": "느림 (+17.63 WPM 부족)",
          "score": 82.37
        },
        {
          "start": 21.33,
          "end": 40.56,
          "wpm": 110.85,
          "feedback": "느림 (+19.15 WPM 부족)",
          "score": 80.85
        },
        {
          "start": 41.87,
          "end": 58.23,
          "wpm": 124.24,
          "feedback": "적절한 속도입니다.",
          "score": 94.24
        }
      ],
      "overall_score": 85.82
    },
    "volume_analysis": {
      "segment_data": [
        {
          "time_stamps": [
            0.46,
            20.18
          ],
          "rms": 65.74,
          "db": 57.38
        },
        {
          "time_stamps": [
            21.33,
            40.56
          ],
          "rms": 67.93,
          "db": 58.47
        },
        {
          "time_stamps": [
            41.87,
            58.23
          ],
          "rms": 73.26,
          "db": 61.42
        }
      ],
      "mean_rms": 68.98,
      "mean_db": 59.09
    },
    "volume_evaluation": {
      "segment_evaluations": [
        {
          "time_stamps": [
            0.46,
            20.18
          ],
          "db": 57.38,
          "feedback": "약간 조용함 (-2.62 dB 부족)",
          "score": 87.38
        },
        {
          "time_stamps": [
            21.33,
            40.56
          ],
          "db": 58.47,
          "feedback": "약간 조용함 (-1.53 dB 부족)",
          "score": 88.47
        },
        {
          "time_stamps": [
            41.87,
            58.23
          ],
          "db": 61.42,
          "feedback": "적절한 음량입니다.",
          "score": 91.42
        }
      ],
      "overall_score": 89.09
    },
    "nonverbal_analysis": [
      {
        "sample_number": 1,
        "time_range": "0.00s ~ 10.00s",
        "frame_range": "0 ~ 49",
        "frame_dir": "frame_0",
        "is_normal": false,
        "wrist_distance": 133.73,
        "top_classes": [
          {
            "class": "자세(비스듬히)",
            "probability": 99.35
          },
          {
            "class": "자세(좌우흔들기)",
            "probability": 0.54
          },
          {
            "class": "손동작(얼굴)",
            "probability": 0.11
          }
        ]
      },
      {
        "sample_number": 2,
        "time_range": "10.00s ~ 20.00s",
        "frame_range": "50 ~ 99",
        "frame_dir": "frame_50",
        "is_normal": true
      },
      {
        "sample_number": 3,
        "time_range": "20.00s ~ 30.00s",
        "frame_range": "100 ~ 149",
        "frame_dir": "frame_100",
        "is_normal": true
      },
      {
        "sample_number": 4,
        "time_range": "30.00s ~ 40.00s",
        "frame_range": "150 ~ 199",
        "frame_dir": "frame_150",
        "is_normal": false,
        "wrist_distance": 119.82,
        "top_classes": [
          {
            "class": "팔동작(무의미반동)",
            "probability": 83.26
          },
          {
            "class": "머리동작(고개흔들기)",
            "probability": 12.87
          },
          {
            "class": "손동작(몸긁기)",
            "probability": 3.87
          }
        ]
      },
      {
        "sample_number": 5,
        "time_range": "40.00s ~ 50.00s",
        "frame_range": "200 ~ 249",
        "frame_dir": "frame_200",
        "is_normal": true
      },
      {
        "sample_number": 6,
        "time_range": "50.00s ~ 58.23s",
        "frame_range": "250 ~ 291",
        "frame_dir": "frame_250",
        "is_normal": false,
        "wrist_distance": 126.19,
        "top_classes": [
          {
            "class": "머리동작(좌우흔들기)",
            "probability": 81.74
          },
          {
            "class": "자세(비스듬히)",
            "probability": 15.32
          },
          {
            "class": "머리동작(숙이기)",
            "probability": 2.94
          }
        ]
      }
    ]
  };
  
  // 발표 ID와 분석 데이터를 매핑하는 맵 객체
  export interface AnalysisDataMap {
    [key: string]: PresentationAnalysis;
  }
  
  export const mockAnalysisDataMap: AnalysisDataMap = {
    "pres-001": presentation1,
    "pres-002": presentation2,
    "pres-003": presentation3
  };
  
  /**
   * 차트 데이터 형식으로 변환하는 유틸리티 함수들
   * AnalysisDashboardPage에서 사용할 수 있는 형식으로 변환
   */
  
  // 속도 데이터를 차트용 형식으로 변환
  export function getPaceChartData(presentationData: PresentationAnalysis): ChartDataPoint[] {
    if (!presentationData || !presentationData.speaking_speed) {
      return [];
    }
  
    return presentationData.speaking_speed.segment_wpm.map(segment => ({
      time: formatTimeStamp(segment.start),
      wpm: segment.wpm
    }));
  }
  
  // 음량 데이터를 차트용 형식으로 변환
  export function getVolumeChartData(presentationData: PresentationAnalysis): ChartDataPoint[] {
    if (!presentationData || !presentationData.volume_analysis) {
      return [];
    }
  
    return presentationData.volume_analysis.segment_data.map(segment => ({
      time: formatTimeStamp(segment.time_stamps[0]),
      db: segment.db
    }));
  }
  
  // 시간 형식 변환 (초 -> MM:SS 형식)
  function formatTimeStamp(seconds: number): string {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }