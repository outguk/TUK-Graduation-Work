# 메인 실행 파일
import subprocess
import os
import shutil
import importlib

# 📌 프로젝트 루트 디렉토리 (main.py가 있는 곳)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 📌 주요 디렉토리 경로
MMACTION2_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "mmaction2"))
DATA_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "data"))
VIDEOS_DIR = os.path.abspath(os.path.join(DATA_DIR, "videos"))
FRAMES_DIR = os.path.abspath(os.path.join(DATA_DIR, "frames"))
RAW_DATA_DIR = os.path.abspath(os.path.join(PROJECT_ROOT,'data','frames'))
video_file_path = os.path.abspath(os.path.join(VIDEOS_DIR, "test.mp4"))

os.chdir(PROJECT_ROOT)
print("현재 실행 디렉토리:", os.getcwd())


# 영상 저장 확인
if not os.path.exists(VIDEOS_DIR):
    print(f"ERROR: {VIDEOS_DIR} 폴더가 존재하지 않습니다. 영상 파일을 넣어주세요")
    exit(1)

print("\n Step 1: 영상에서 프레임 추출 (extract_frames.py 실행)\n")
subprocess.run(["python", os.path.join(MMACTION2_DIR, "script", "extract_frames.py"), video_file_path],cwd=PROJECT_ROOT, check=True)

# 키포인트 검출 실행
print("\n Step 2: crop_image_keypoints.py 실행\n")
subprocess.run(["python", os.path.join(MMACTION2_DIR, "mmpose", "crop_image_keypoints.py")],cwd=PROJECT_ROOT, check=True)

# JSON을 PKL로 변환
print("\n Step 3: jsontopkl.py 실행\n")
subprocess.run(["python", os.path.join(MMACTION2_DIR, "detect_pose", "cal_movement.py")],cwd=PROJECT_ROOT, check=True)

# MMACTION2 폴더 내에서 명령어 실행 (테스트 수행)
print("\n Step 4: ST-GCN++ 모델 테스트 실행 (test.py 실행)\n")
test_command = [
    "python", "tools/test.py",
    "configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d.py",
    "work_dirs/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d/best_acc_top1_epoch_27.pth",
    "--dump", "../data/test_results.pkl"
]

subprocess.run(test_command, cwd=MMACTION2_DIR, check=True)

# 최종 결과 확인
print("\n Step 5: 모델 예측 결과 확인 (checkresult.py 실행)\n")
# subprocess.run(["python", os.path.join(MMACTION2_DIR, "2s", "checkresult.py")],cwd=PROJECT_ROOT, check=True)
import sys
CHECKRESULT_DIR = os.path.join(MMACTION2_DIR, "2s")
sys.path.insert(0, CHECKRESULT_DIR)  # 가장 우선순위로 설정

# 🔹 기존에 잘못 import된 checkresult 제거 (중복 import 방지)
if "checkresult" in sys.modules:
    del sys.modules["checkresult"]

# 🔹 강제로 checkresult import
import checkresult
importlib.reload(checkresult)  # checkresult.py 재로드 (필수)

result_file = os.path.join(DATA_DIR, "test_results.pkl")
keypoints_pkl_file = os.path.join(DATA_DIR, "keypoints/results.pkl")
output_json_file = os.path.join(DATA_DIR, "inference_results.json")

result_dict = checkresult.main(result_file, keypoints_pkl_file, output_json_file)

# result_dict 
# Dict 출력
print(result_dict)
print("\n모든 과정 완료\n")

"""
result_dict 출력 예시:

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
"""