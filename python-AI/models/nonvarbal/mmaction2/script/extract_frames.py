import cv2
import os
import sys

def video_to_frames(video_path, output_dir, file_format='jpg'):
    """
    영상을 2초 간격, 각 구간에서 10개의 프레임 추출하여 이미지 파일로 저장
    
    Args:
        video_path (str): 비디오 파일 경로.
        output_dir (str): 프레임을 저장할 디렉토리 경로.
        file_format (str): 저장할 이미지 파일 형식 ('jpg' 또는 'png').
    
    Returns:
        int: 저장된 프레임의 총 개수.
    """

    # sys.argv 처리는 필요에 따라 조정합니다.
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        raise ValueError("❌ ERROR: 비디오 파일 경로가 제공되지 않았습니다.")
    
    # 스크립트 경로와 프로젝트 루트 경로 계산
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))
    
    # video_path 와 output_dir 을 절대 경로로 보정
    video_path = os.path.abspath(os.path.join(project_root, video_path))
    output_dir = os.path.abspath(os.path.join(project_root, "data/frames"))  # data/frames 기준

    print(f" [DEBUG] 실행 디렉토리: {script_dir}")
    print(f" [DEBUG] 비디오 파일 경로: {video_path}")
    print(f" [DEBUG] 프레임 저장 기본 경로: {output_dir}")

    # ✅ 비디오 파일 확인 로그 추가
    print(f"\n📂 [DEBUG] 비디오 파일 경로 확인: {video_path}")

    # 업로드된 비디오 파일 이름(확장자 제거)으로 된 폴더 생성
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_filename)
    os.makedirs(video_output_dir, exist_ok=True)
    
    print(f" [DEBUG] 최종 프레임 저장 경로: {video_output_dir}")

    # 비디오 파일 열기
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")
    print("\n✅ 비디오 파일이 정상적으로 열렸습니다!")

    # 비디오의 FPS(초당 프레임 수) 확인
    fps = video.get(cv2.CAP_PROP_FPS)
    print(f"영상의 FPS: {fps}")

    # 2초 동안의 총 프레임 수
    block_duration_frames = int(round(2.0 * fps))  # 2초

    # 각 2초 구간 내에서 균등하게 10개의 프레임 인덱스 계산
    sample_indices = [int(round(i * (block_duration_frames - 1) / 9)) for i in range(10)]
    print("각 2초 구간에서 추출할 프레임 인덱스:", sample_indices)

    frame_count = 0  # 전체 프레임 순번 (영상 내)
    saved_count = 0  # 실제로 저장된 프레임 개수 (파일명 뒤 번호로 사용)

    success, frame = video.read()  # 첫 번째 프레임 읽기
    while success:
        # 현재 프레임이 속한 블록 내 인덱스 (0 ~ block_duration_frames-1)
        current_index_in_block = frame_count % block_duration_frames

        # 샘플링 인덱스에 해당하면 이미지 파일로 저장
        if current_index_in_block in sample_indices:
            frame_filename = os.path.join(video_output_dir, f"frame_{saved_count}.{file_format}")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        # 다음 프레임 읽기
        success, frame = video.read()
        frame_count += 1

    video.release()  # 비디오 객체 해제
    print(f"\n총 {saved_count}개의 프레임이 '{video_output_dir}' 경로에 저장되었습니다.")
    return saved_count

# 테스트 실행용 예시 (직접 실행 시)
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../../../../"))
    
    # 예시 비디오, 실제 환경에 맞게 수정
    video_path = os.path.join(project_root, 'uploaded_videos/test.mp4')
    output_dir = os.path.join(project_root, 'models/nonvarval/data/frames')
    file_format = 'jpg'

    saved_frames = video_to_frames(video_path, output_dir, file_format)
    print(f"{saved_frames}개의 프레임이 저장되었습니다.")
