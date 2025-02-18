# nonverbal/data/videos에 test.mp4 저장

# nonverbal 폴더에서 main.py 실행

1. 영상 프레임 추출 (mmaction2/script/extract_frames.py)

2. keypoint 추출 (mmaction2/mmpose/crop_image_keypoints.py)

3. keypoints json파일 모델 입력 데이터(.pkl) 변환 (mmaction2/detect_pose/cal_movement.py)

- 이전 프레임과 현재 프레임의 movement를 계산하여 30보다 작을경우 모델 입력 데이터에서 제외.

4. STGCN++ 모델 예측 수행 (mmaction2/tool/test.py)

5. 예측 결과 확인 (mmaction2/2s/checkresult.py)

- movement 계산에서 제외된 프레임은 별도 출력
- top1의 예측 확률이 85%미만일 경우 정상행동으로 출력 후 top1~3의 확률 출력

# 예측 결과 출력 예시

- 샘플 6 (영상 구간: 10.00s ~ 11.80s, 프레임: 50 ~ 59, frame_dir: frame_50):
  정상 행동
  1.  손동작(얼굴) (51.12%)
  2.  머리동작(좌우흔들기) (48.54%)
  3.  자세(비스듬히) (0.32%)

⚠ [정상 행동 구간] 60 ~ 69 프레임 건너뜀

샘플 7 (영상 구간: 14.00s ~ 15.80s, 프레임: 70 ~ 79, frame_dir: frame_70):

1.  자세(좌우흔들기) (100.00%)
2.  팔동작(무의미반동) (0.00%)
3.  손동작(머리) (0.00%)
