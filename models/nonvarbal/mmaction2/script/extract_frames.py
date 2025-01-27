import cv2
import os

def video_to_frames(video_path, output_dir, file_format='jpg'):
    """
    영상을 초당 30프레임으로 잘라 이미지 파일로 저장.

    Args:
        video_path (str): 비디오 파일 경로.
        output_dir (str): 프레임을 저장할 디렉토리 경로.
        file_format (str): 저장할 이미지 파일 형식 ('jpg' 또는 'png').

    Returns:
        int: 저장된 프레임의 총 개수.
    """
    # 비디오 파일명에서 확장자 제거하여 폴더 이름 생성
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    
    # 저장 디렉토리 생성
    os.makedirs(video_output_dir, exist_ok=True)

    video = cv2.VideoCapture(video_path)  # 비디오 파일 열기
    if not video.isOpened():
        raise ValueError(f"비디오 파일을 열 수 없습니다: {video_path}")

    # 비디오의 FPS(초당 프레임 수) 확인
    fps = int(video.get(cv2.CAP_PROP_FPS))
    print(f"영상의 FPS: {fps}")

    frame_count = 0  # 전체 프레임 수
    saved_count = 0  # 저장된 프레임 수
    success, frame = video.read()  # 첫 번째 프레임 읽기

    while success:
        # 모든 프레임 저장 (초당 FPS만큼 저장)
        frame_filename = os.path.join(video_output_dir, f"frame_{frame_count}.{file_format}")
        cv2.imwrite(frame_filename, frame)  # 프레임 저장
        saved_count += 1

        success, frame = video.read()  # 다음 프레임 읽기
        frame_count += 1

    video.release()  # 비디오 객체 해제
    print(f"총 {saved_count}개의 프레임이 {video_output_dir}에 저장되었습니다.")
    return saved_count


if __name__ == "__main__":
    video_path = '../data/videos/demo.mp4' # 비디오 파일 경로
    output_dir = '../data/frames/'          # 프레임 저장 경로
    file_format = 'jpg'                     # 저장 파일 형식 ('jpg' 또는 'png')

    saved_frames = video_to_frames(video_path, output_dir, file_format)
    print(f"{saved_frames}개의 프레임이 저장되었습니다.")
