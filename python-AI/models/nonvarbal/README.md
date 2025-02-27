# nonverbal/data/videos에 테스트 영상 test.mp4 저장

# nonverbal 폴더에서 main.py 실행

# -- noverbal 실행 순서 및 설명 --

1. 영상 프레임 추출 (mmaction2/script/extract_frames.py)

   - 비디오 파일 열기 (OpenCV 사용)
   - FPS(초당 프레임 수) 확인
   - 2초 단위로 일정 개수의 프레임을 추출할 블록 설정
   - 각 블록에서 10개 프레임을 균등하게 선택
   - 선택한 프레임들을 이미지 파일로 저장
   - 비디오 객체 해제 및 저장된 총 프레임 개수 반환

2. keypoints 추출 (mmaction2/mmpose/crop_image_keypoints.py)

   - 입력된 프레임 디렉토리에서 모든 이미지 파일을 읽음
   - 이미지를 중앙 크롭 (가로가 길 경우 정사각형으로 변환)
   - 크롭된 이미지를 MMPose 모델(Hrnet)에 입력하여 키포인트 추출
   - 키포인트를 JSON 파일로 저장
   - (옵션) 키포인트를 시각화하여 저장

3. keypoints json파일 모델 입력 데이터(.pkl) 변환 (mmaction2/detect_pose/cal_movement.py)

   - 키포인트 JSON 파일을 불러와 정렬
   - 10개 프레임 단위로 묶어서 비교 (frames_per_annotation=10)
   - 각 프레임 간 키포인트 좌표 변화량 계산
   - 만약 변화량이 movement_threshold(30) 보다 크면 모델 입력 데이터로 저장
   - 결과를 .pkl`(바이너리) 파일로 변환하여 저장

4. STGCN++ 모델 예측 수행 (mmaction2/tool/test.py)

   - 명령행 인자를 통해 설정파일(config), 체크포인트(checkpoint) 로드
   - MMAction2의 Runner를 초기화하여 모델 실행
   - data/keypoints/results.pkl을 입력받아 모델 테스트 진행
   - 예측 결과를 test_results.pkl에 저장

5. 예측 결과 확인 (mmaction2/2s/checkresult.py)

   - test_results.pkl (예측 결과) & results.pkl (입력 데이터) 불러오기
   - 각 프레임별 예측 확률이 높은 Top-3 행동을 출력
   - 만약 정상 행동(움직임이 적음)이면 별도 처리

   - checkresult 딕셔너리 형식 결과 반환(JSON 파일과 같은 내용)

- movement 계산에서 제외된 프레임은 별도 출력
- top1의 예측 확률이 85%미만일 경우 정상행동으로 출력 후 top1~3의 확률 출력

# 예측 결과 출력 예시

샘플 6 (영상 구간: 10.00s ~ 11.80s, 프레임: 50 ~ 59, frame_dir: frame_50):

정상 행동 (확률 85% 미만)

1.  손동작(얼굴) (51.12%)
2.  머리동작(좌우흔들기) (48.54%)
3.  자세(비스듬히) (0.32%)

⚠ [정상 행동 구간] 60 ~ 69 프레임 건너뜀

샘플 7 (영상 구간: 14.00s ~ 15.80s, 프레임: 70 ~ 79, frame_dir: frame_70):

1.  자세(좌우흔들기) (100.00%)
2.  팔동작(무의미반동) (0.00%)
3.  손동작(머리) (0.00%)

# JSON 파일 속성, 타입, 설명

```
- sample_number   : int      - 샘플 번호 (1부터 시작)
- time_range      : string   - 분석된 영상 구간 (초 단위, 시작시간 ~ 종료시간)
- frame_range     : string   - 분석된 프레임 범위 (예: 0 ~ 9)
- frame_dir	      : string   - 분석된 프레임이 저장된 디렉토리명 (예: frame_0)
- is_normal	      : bool     - 정상 행동 여부 (true: 정상 행동, false: 이상 행동)
- top_classes     : list     - (이상 행동일 경우) 예측 확률이 높은 상위 3개 클래스
   - class	        : string   - 예측된 행동 클래스 (예: 손동작(얼굴))
   - probability    : float    - 해당 행동의 예측 확률 (%)
```

# JSON 데이터 출력 예시

```
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
```

# checkresult.py의 리턴값 예시

```
{
    1: {
        "sample_number": 1,
        "time_range": "0.00s ~ 2.00s",
        "frame_range": "0 ~ 9",
        "frame_dir": "frame_0",
        "is_normal": False,
        "top_classes": [
            {"class": "팔동작(무의미반동)", "probability": 88.5},
            {"class": "머리동작(고개흔들기)", "probability": 5.2},
            {"class": "자세(비비꼬기)", "probability": 3.1}
        ]
    },
    2: {
        "sample_number": 2,
        "time_range": "2.00s ~ 4.00s",
        "frame_range": "10 ~ 19",
        "frame_dir": "frame_10",
        "is_normal": True
    }
}
```
