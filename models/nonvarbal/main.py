# 메인 실행 파일
import subprocess
import os
import shutil

# 📌 프로젝트 루트 디렉토리 (main.py가 있는 곳)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 📌 주요 디렉토리 경로
MMACTION2_DIR = os.path.join(PROJECT_ROOT, "mmaction2")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
FRAMES_DIR = os.path.join(DATA_DIR, "frames")
RAW_DATA_DIR = os.path.join(PROJECT_ROOT,'data','frames')
video_file_path = os.path.join(VIDEOS_DIR, "test.mp4")

# 영상 저장 확인
if not os.path.exists(VIDEOS_DIR):
    print(f"ERROR: {VIDEOS_DIR} 폴더가 존재하지 않습니다. 영상 파일을 넣어주세요")
    exit(1)

print("\n🚀 Step 1: 영상에서 프레임 추출 (extract_frames.py 실행)")
subprocess.run(["python", os.path.join(MMACTION2_DIR, "script", "extract_frames.py"), video_file_path], check=True)

# 키포인트 검출 실행
print("\n🚀 Step 2: crop_image_keypoints.py 실행")
subprocess.run(["python", os.path.join(MMACTION2_DIR, "mmpose", "crop_image_keypoints.py")], check=True)

# JSON을 PKL로 변환
print("\n🚀 Step 3: jsontopkl.py 실행")
subprocess.run(["python", os.path.join(MMACTION2_DIR, "detect_pose", "cal_movement.py")], check=True)

# MMACTION2 폴더 내에서 명령어 실행 (테스트 수행)
print("\n🚀 Step 4: ST-GCN++ 모델 테스트 실행 (test.py 실행)")
test_command = [
    "python", "tools/test.py",
    "configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d.py",
    "work_dirs/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d/best_acc_top1_epoch_27.pth",
    "--dump", "../data/test_results.pkl"
]

subprocess.run(test_command, cwd=MMACTION2_DIR, check=True)

# 최종 결과 확인
print("\n Step 5: 모델 예측 결과 확인 (checkresult.py 실행)")
subprocess.run(["python", os.path.join(MMACTION2_DIR, "2s", "checkresult.py")], check=True)

print("\n모든 과정 완료")
