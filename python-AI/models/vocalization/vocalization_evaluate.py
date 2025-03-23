# 발성 평가 코드
import logging
import sys
import os
import moviepy.editor as mp
import asyncio

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def evaluate_speaking_speed(metrics: dict) -> float:
    """
    analyze_speaking_speed 함수가 반환한 결과(metrics) 내의 segment_wpm 데이터를 사용하여
    각 구간별로 말하기 속도를 평가합니다.
    
    기준 예시:
      - 이상적인 말하기 속도: 130 ~ 150 WPM
      - 구간별로 벗어난 정도에 따라 점수를 감점하고, '너무 빠름' 또는 '너무 느림' 피드백을 제공합니다.
      - 10 WPM 차이마다 10점씩 감점 (최소 0점, 최대 100점)
      
    매개변수:
      metrics (dict): analyze_speaking_speed 함수가 반환한 결과
                      예시: {
                        "overall_wpm": 140,
                        "total_spoken_time": 35.5,
                        "segment_wpm": [
                           {"start": 0.0, "end": 5.0, "wpm": 125, "words": "..." },
                           {"start": 5.0, "end": 10.0, "wpm": 165, "words": "..." },
                           ...
                        ]
                      }
    
    반환:
      dict: {
             "segment_evaluations": [각 구간별 평가 결과],
             "overall_score": 종합 평가 점수
            }
    """

     # 이상적인 말하기 속도 범위 설정 (분당 단어 수)
    ideal_min = 130
    ideal_max = 150

    # 각 구간별 평가 결과를 저장할 리스트를 초기화
    segment_evaluations = []
    
    # metrics 딕셔너리에서 구간별 WPM 데이터를 가져옴
    segments = metrics.get("segment_wpm", [])
    logging.info(f"총 {len(segments)} 구간에 대해 평가를 시작합니다.")
    
    # 각 발화 구간(segment)에 대해 평가를 진행
    for seg in segments:
        wpm = seg.get("wpm", 0)

        # 피드백 문자열과 감점(penalty) 값을 초기화
        feedback = ""
        penalty = 0
        
        # 만약 구간의 WPM이 이상적인 최소값보다 낮으면 '느림'으로 평가
        if wpm < ideal_min:
            diff = ideal_min - wpm # diff : 오차 
            penalty = (diff / 10) * 10  # 10 WPM 차이당 10점 감점
            feedback = f"느림 (+{round(diff,2)} WPM 부족)"
        # 만약 구간의 WPM이 이상적인 최대값보다 높으면 '빠름'으로 평가
        elif wpm > ideal_max:
            diff = wpm - ideal_max
            penalty = (diff / 10) * 10
            feedback = f"빠름 (+{round(diff,2)} WPM 초과)"
        # 이상적인 범위 내에 있으면 적정 속도로 평가
        else:
            feedback = "적정 속도"
        
        # 점수는 100점에서 감점, 최소 0점 보장 
        score = max(100 - penalty, 0)

        # 현재 구간의 평가 결과를 딕셔너리로 구성하여 리스트에 추가
        segment_evaluations.append({
            "start": seg.get("start"),
            "end": seg.get("end"),
            "wpm": wpm,
            "feedback": feedback,
            "score": round(score, 2)
        })
    
    # 전체 점수는 각 구간 점수의 평균으로 산출 (구간이 없으면 0 처리)
    if segment_evaluations:
        overall_score = sum(item["score"] for item in segment_evaluations) / len(segment_evaluations)
    else:
        overall_score = 0
    
    # 평가 결과를 딕셔너리 형태로 반환
    return {
        "segment_evaluations": segment_evaluations,
        "overall_score": round(overall_score, 2)
    }

# 음량 평가 함수

def evaluate_volume(volume_metrics: dict) -> float:
    """
    analyze_volume 함수가 반환한 결과(volume_metrics) 내의 segment_data를 사용하여
    각 구간별 음량을 평가하는 함수입니다.
    
    평가 기준 (예시):
      - 이상적인 음량 범위: ideal_min ~ ideal_max dB
      - 각 구간의 dB 값이 기준 범위를 벗어난 경우, 벗어난 정도에 따라 감점합니다.
      - 5 dB 차이마다 10점씩 감점 (최소 0점, 최대 100점)
      - 각 구간에 대해 '조용함' 혹은 '시끄러움'이라는 피드백을 제공합니다.
      - 모든 구간의 점수를 평균하여 전체 음량 평가 점수를 산출합니다.

    매개변수:
      volume_metrics (dict): analyze_volume 함수의 결과값. 예시:
                              {
                                "segment_data": [
                                  {"time_stamps": (start, end), "rms": value, "db": value},
                                  ...
                                ],
                                "mean_rms": value,
                                "mean_db": value
                              }
                              
    반환:
      dict: 평가 결과 딕셔너리, 예:
            {
                "segment_evaluations": [각 구간별 평가 결과 리스트],
                "overall_score": 전체 평균 평가 점수
            }
    """
    # 이상적인 음량 범위 설정
    ideal_min = 60
    ideal_max = 70
    
    # 각 구간별 평가 결과를 저장할 리스트 초기화
    segment_evaluations = []
    
    # volume_metrics 딕셔너리에서 각 구간의 음량 데이터 리스트를 가져옵니다.
    segments = volume_metrics.get("segment_data", [])
    logging.info(f"총 {len(segments)} 음량 구간에 대해 평가를 시작합니다.")

    # 각 음량 구간에 대해 반복 평가 수행
    for seg in segments:
        # 구간의 시간 정보와 dB 값을 추출 (rms 값은 필요에 따라 추가 사용 가능)
        time_stamps = seg.get("time_stamps", (None, None))
        db = seg.get("db", 0)
        logging.info(f"구간 시간: {time_stamps}, dB: {db}")
        
        # 피드백 문자열과 감점(penalty) 값을 초기화
        feedback = ""
        penalty = 0
        
        # 기준보다 음량(dB)이 낮은 경우: 조용함
        if db < ideal_min:
            diff = ideal_min - db  # 부족한 dB 차이 계산
            penalty = (diff / 10) * 10  # 5 dB 차이마다 10점 감점
            feedback = f"조용함 (-{round(diff,2)} dB 부족)"
        # 기준보다 음량이 높은 경우: 시끄러움
        elif db > ideal_max:
            diff = db - ideal_max  # 초과한 dB 차이 계산
            penalty = (diff / 10) * 10  # 5 dB 차이마다 10점 감점
            feedback = f"시끄러움 (+{round(diff,2)} dB 초과)"
        # 음량이 이상적인 범위 내에 있는 경우
        else:
            feedback = "적정 음량"
            logging.info("음량이 적정 범위 내에 있어 감점 없음.")
        
        # 100점에서 감점된 점수를 계산 (최소 0점 보장)
        score = max(100 - penalty, 0)
        logging.info(f"해당 구간 점수: {score}")
        
        # 현재 구간의 평가 결과를 딕셔너리로 구성하여 리스트에 추가 
        segment_evaluations.append({
            "time_stamps": time_stamps,
            "db": db,
            "feedback": feedback,
            "score": round(score, 2)
        })
    
    # 모든 구간의 점수를 평균하여 전체 평가 점수를 산출
    if segment_evaluations:
        overall_score = sum(item["score"] for item in segment_evaluations) / len(segment_evaluations)
    else:
        overall_score = 0
    logging.info(f"전체 음량 평가 점수 (평균): {overall_score}")
    
    # 평가 결과를 딕셔너리 형태로 반환
    return {
        "segment_evaluations": segment_evaluations,
        "overall_score": round(overall_score, 2)
    }
