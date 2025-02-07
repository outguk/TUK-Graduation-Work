# 음량 분석 테스트 코드

import os
import sys
import pytest

# 프로젝트 루트를 sys.path에 추가(경로 검색을 쉽게 하기)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# 디버깅 출력
print("Correct project root added to sys.path:", project_root)
print("sys.path:", sys.path)

# 필요한 모듈 임포트
from models.vocalization.vocalization_analysis import analyze_volume

def test_analyze_volume():
    # 테스트용 동영상 파일 경로 및 출력 텍스트 파일 경로
    audio_file_path = os.path.abspath("python-AI/tests/test_video/test1_audio.wav")  # 실제 테스트용 파일 필요

    # Step 1: 동영상 파일 확인
    assert os.path.exists(audio_file_path), f"Test video file does not exist: {audio_file_path}"

    # Step 2: 함수 실행 및 텍스트 추출
    result = analyze_volume(audio_file_path)

    # Step 3: 결과 검증
    # (1) 반환된 결과가 비어 있는 지 확인
    assert result, "result is empty."
    assert "time_stamps" in result, "result does not contain the required time_stamps."
    assert "rms_values" in result, "result does not contain the required rms_values"
    assert "db_values" in result, "result does not contain the required db_values."

    # Step 4: 분석 결과 확인
    # print(result)

    # 테스트 성공