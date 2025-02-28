# 비언어적 요소 분석 코드
import subprocess
import os
import sys

def video_nonverbal_analysis(video_file_path: str) -> dict:
    """
    주어진 비디오 파일 경로에 대해 프레임 추출, 키포인트 검출, JSON→PKL 변환,
    ST-GCN++ 모델 테스트, 예측 결과 확인 등의 과정을 순차적으로 실행한다.
    """
    # 프로젝트 루트 디렉토리 (video_analysis.py와 main.py가 같은 위치에 있다고 가정)
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    
    # 주요 디렉토리 경로 설정
    MMACTION2_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "mmaction2"))
    DATA_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "data"))
    # VIDEOS_DIR = os.path.abspath(os.path.join(DATA_DIR, "videos"))
    
    # 작업 디렉토리를 PROJECT_ROOT로 변경
    os.chdir(PROJECT_ROOT)
    print("현재 실행 디렉토리:", os.getcwd())
    
    try:
        # 영상 저장 폴더 확인
        if not os.path.exists(video_file_path):
            raise Exception(f"ERROR: {video_file_path} 폴더가 존재하지 않습니다. 영상 파일을 넣어주세요")
        
        # Step 1: 영상에서 프레임 추출 (extract_frames.py 실행)
        print("\n🚀 Step 1: 영상에서 프레임 추출 (extract_frames.py 실행)")
        subprocess.run(
            ["python", os.path.join(MMACTION2_DIR, "script", "extract_frames.py"), video_file_path],
            cwd=PROJECT_ROOT, check=True
        )
        
        # Step 2: 키포인트 검출 실행 (crop_image_keypoints.py 실행)
        print("\n🚀 Step 2: crop_image_keypoints.py 실행")
        subprocess.run(
            ["python", os.path.join(MMACTION2_DIR, "mmpose", "crop_image_keypoints.py")],
            cwd=PROJECT_ROOT, check=True
        )
        
        # Step 3: JSON을 PKL로 변환 (cal_movement.py 실행)
        print("\n🚀 Step 3: jsontopkl.py 실행")
        subprocess.run(
            ["python", os.path.join(MMACTION2_DIR, "detect_pose", "cal_movement.py")],
            cwd=PROJECT_ROOT, check=True
        )
        
        # Step 4: ST-GCN++ 모델 테스트 실행 (test.py 실행)
        print("\n🚀 Step 4: ST-GCN++ 모델 테스트 실행 (test.py 실행)")
        test_command = [
            "python", "tools/test.py",
            "configs/skeleton/stgcnpp/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d.py",
            "work_dirs/stgcnpp_8xb16-joint-motion-u100-80e_ntu60-xsub-keypoint-2d/best_acc_top1_epoch_27.pth",
            "--dump", "../data/test_results.pkl"
        ]
        subprocess.run(test_command, cwd=MMACTION2_DIR, check=True)
        
        # Step 5: 모델 예측 결과 확인 (checkresult.py 실행) 및 분석 결과 반환
        print("\n🚀 Step 5: 모델 예측 결과 확인 (checkresult.py 실행)")
        checkresult_dir = os.path.join(MMACTION2_DIR, "2s")
        sys.path.insert(0, checkresult_dir)
        print("📌 sys.path:", sys.path)

        import checkresult

        result_file = os.path.join(DATA_DIR, "test_results.pkl")
        keypoints_pkl_file = os.path.join(DATA_DIR, "keypoints/results.pkl")
        output_json_file = os.path.join(DATA_DIR, "inference_results.json")

        result = checkresult.main(result_file, keypoints_pkl_file, output_json_file)
        
        
        print("\n모든 과정 완료")
    
    # 오류 발생 시
    except subprocess.CalledProcessError as e:
        result = {"status": "error", "message": f"서브프로세스 실행 중 오류 발생: {e}"}
    except Exception as e:
        result = {"status": "error", "message": str(e)}
    
    return result

# 직접 모듈을 실행할 경우 테스트용 코드
if __name__ == "__main__":
    # data/videos 폴더 내의 test.mp4 파일 경로 설정
    sample_video = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "videos")),
        "test.mp4"
    )
    analysis_result = video_nonverbal_analysis(sample_video)
    print("분석 결과:", analysis_result)
