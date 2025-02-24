import json

# json으로 저장된 결과를 출력하는 코드
# JSON 파일 경로 nonverbal/data/inference_results.json

# JSON 파일 경로
json_file = "inference_results.json"

# JSON 파일 로드
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# JSON 데이터 출력 
print(json.dumps(data, indent=4, ensure_ascii=False))

# JSON 속성, 데이터 타입, 설명 
"""
sample_number	: int	    - 샘플 번호 (1부터 시작)
time_range      : string	- 분석된 영상 구간 (초 단위, 시작시간 ~ 종료시간)
frame_range     : string	- 분석된 프레임 범위 (예: "0 ~ 9")
frame_dir	    : string	- 분석된 프레임이 저장된 디렉토리명 (예: frame_0)
is_normal	    : bool	    - 정상 행동 여부 (true: 정상 행동, false: 이상 행동)
top_classes     : list   	- (이상 행동일 경우) 예측 확률이 높은 상위 3개 클래스
    class	       : string	   - 예측된 행동 클래스 (예: 손동작(얼굴))
    probability    : float	   - 해당 행동의 예측 확률 (%)
"""
# JSON 데이터 출력 예시

"""
{
    "results": [
        {
            "sample_number": 1,
            "time_range": "0.00s ~ 1.80s",
            "frame_range": "0 ~ 9",
            "frame_dir": "frame_0",
            "is_normal": false,
            "top_classes": [ 
                {
                    "class": "자세(비비꼬기)",
                    "probability": 100.0
                },
                {
                    "class": "자세(비스듬히)",
                    "probability": 0.0
                },
                {
                    "class": "손동작(머리)",
                    "probability": 0.0
                }
            ]
        },
        {
            "sample_number": 2,
            "time_range": "2.00s ~ 3.80s",
            "frame_range": "10 ~ 19",
            "frame_dir": "frame_10",
            "is_normal": true    
        },
        {
            "sample_number": 3,
            "time_range": "4.00s ~ 5.80s",
            "frame_range": "20 ~ 29",
            "frame_dir": "frame_20",
            "is_normal": false,
            "top_classes": [
                {
                    "class": "손동작(머리)",
                    "probability": 100.0
                },
                {
                    "class": "팔동작(무의미반동)",
                    "probability": 0.0
                },
                {
                    "class": "머리동작(고개흔들기)",
                    "probability": 0.0
                }
            ]
        },   
"""