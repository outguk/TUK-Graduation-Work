import os
import sys
import pytest
import warnings
import json

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# 디버깅 출력
print("Correct project root added to sys.path:", project_root)
print("sys.path:", sys.path)

# 필요한 모듈 임포트
from models.vocalization.vocalization_analysis import transcribe_audio

async def test_transcribe_audio():
    # 테스트용 동영상 파일 경로 및 출력 텍스트 파일 경로
    video_file = os.path.abspath("tests/test_video/test2_preprocessed.wav")  # 실제 테스트용 파일 필요(전처리 완료 오디오 파일)
    output_script_file = os.path.abspath("tests/output/test_transcription.txt")  # 저장 경로 설정

    # Step 1: 동영상 파일 확인
    assert os.path.exists(video_file), f"Test video file does not exist: {video_file}"

    # Step 2: 함수 실행 및 텍스트 추출
    transcription = await transcribe_audio(video_file, model_name="base")
    transcription_text = transcription['text']  # 변환된 텍스트 가져오기

    print("Transcription type:", type(transcription_text))
  

    # Step 3: 결과 검증
    # (1) 반환된 텍스트가 비어 있지 않은지 확인
    assert transcription_text.strip(), "Transcription is empty."

    # (2) Whisper가 반환한 텍스트의 일부를 출력 (검증을 위한 참조)
    # print("Transcribed Text:", transcription['text'][:100])

    # Step 4: 텍스트 파일로 저장
    try:
        os.makedirs(os.path.dirname(output_script_file), exist_ok=True)  # 경로가 없으면 생성
        with open(output_script_file, "w", encoding="utf-8") as f:
            # f.write(transcription)  # 변환된 텍스트 저장
            json.dump(transcription, f, ensure_ascii=False, indent=2)
        print(f"Transcription saved to: {output_script_file}")
    except Exception as e:
        print(f"Failed to save transcription: {e}")

    # Step 4: 경고 무시 (선택 사항)
    warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

    '''Whisper 모델은 기본적으로 FP16(16비트 부동소수점) 연산을 사용하려고 시도합니다.
       CPU에서는 FP16 연산을 지원하지 않으므로 FP32(32비트 부동소수점)로 대체합니다.
       -> 최적화 관련(나중에 살펴보기) 현재 로컬 노트북으로는 GPU 사용불가'''
    
    # 테스트 성공