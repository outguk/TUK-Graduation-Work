import os
import sys
import pytest

# 프로젝트 루트를 sys.path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)


from models.vocalization.vocalization_analysis import extract_audio

def test_extract_audio():
    print("")

# 테스트 성공 (wav파일 -> 다른 파일로 하는 것이 좋은 지 유의)