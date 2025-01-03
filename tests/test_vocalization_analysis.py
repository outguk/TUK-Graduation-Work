import pytest
from models.vocalization.vocalization_analysis import extract_audio

def test_analyze_volume():
    # 테스트용 오디오 파일 경로
    test_audio_file = r"C:\Users\dlsrn\OneDrive\바탕 화면\종합설계\Sample\Sample\01.원천데이터\1.언어적\2. A01 고등학생\A01_S01_M_F_08_139_02_WA_MO.mp4"  # 샘플 오디오 파일
    
    # 함수 호출
    result = extract_audio(test_audio_file, output_audio_file="audio.wav")
    
    # 결과 검증
    # assert => 간단한 디버깅 도구, 특정 조건이 True인지 확인하고, 조건이 False일 경우 프로그램 실행을 중단
