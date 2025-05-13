# 발성 분석 코드

import logging
import sys
import os
import moviepy.editor as mp
import whisper
import json
import numpy as np
import matplotlib.pyplot as plt
import asyncio
import math
from pydub import AudioSegment, silence
from collections import Counter
from cmudict import entries as cmu_dict

# 프로젝트 루트를 sys.path에 추가(경로 검색을 쉽게 하기 위해)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# Step 1: Extract Audio from Video
# 동영상에서 오디오 추출 -> 최적화 및 개선 필요
async def extract_audio(video_file, output_audio_file):
    """
    mp4 영상에서 wav 오디오를 추출하는 함수
    
    매개변수:
        video_file : 비디오 파일 경로
        output_audio_file : 추출된 오디오 파일일

    """
    loop = asyncio.get_running_loop()

    def _extract():
        logging.info(f"🎞 Extracting audio from: {video_file}")
        try:
            video = mp.VideoFileClip(video_file)
            video.audio.write_audiofile(output_audio_file)
            # 오디오 추출 완료 메시지 출력
            print(f"Audio extracted to: {output_audio_file}")
            return output_audio_file
        except FileNotFoundError:
            logging.error(f"❌ Video file not found: {video_file}")
            return None

    return await loop.run_in_executor(None, _extract)

# Step 2: Speech-to-Text with Whisper
# 음성을 텍스트로 변환 (Whisper 사용) -> 최적화 및 개선 필요
async def transcribe_audio(audio_file, model_name="base"):
    loop = asyncio.get_running_loop()

    def _transcribe():
        logging.info(f"🧠 Transcribing audio with Whisper: {audio_file}")
        model = whisper.load_model(model_name)  # Whisper 모델 로딩
        transcription = model.transcribe(audio_file)
        logging.info(transcription["text"][:100])
        return transcription     # 오디오 텍스트 변환

    # run_in_executor로 Whisper STT를 스레드에서 실행
    return await loop.run_in_executor(None, _transcribe)

# Step 3: Analyze Speaking Speed (Words per Minute)
# 말하기 속도 분석 (분당 단어 수 계산) -> 최적화 및 개선 필요(무음 구간 추출 부분 모듈화화)
def analyze_speaking_speed(transcription):
    """
    Whisper 대본 세그먼트를 기반으로 발화 속도를 분석하는 함수입니다.
    
    매개변수:
      transcription (dict):
          Whisper가 생성한 대본입니다. "segments" 키가 포함되어 있어야 하며,
          각 세그먼트는 "start", "end", "text" 필드를 포함하는 dict여야 합니다.
          
    반환값:
      dict: 다음 키들을 포함하는 딕셔너리로, 원래 함수의 반환 형식과 일치합니다.
          - "overall_wpm": 전체 단어 수를 (전체 발화 시간(초)에 대해 60을 곱한 값)으로 계산한 전체 발화 속도 (WPM).
          - "total_spoken_time": 모든 세그먼트의 발화 시간 합 (초 단위).
          - "segment_wpm": 각 세그먼트별 발화 속도 정보를 담은 딕셔너리 리스트로, 각 딕셔너리는 다음을 포함합니다.
                "start": 해당 세그먼트의 시작 시간 (초, 소수점 둘째 자리 반올림)
                "end": 해당 세그먼트의 종료 시간 (초, 소수점 둘째 자리 반올림)
                "wpm": 해당 세그먼트의 발화 속도 (소수점 둘째 자리 반올림)
                "words": 해당 세그먼트의 대본 텍스트
          
    예제:
      test4.txt :contentReference[oaicite:0]{index=0}에서 읽어들인 대본을 사용하여 함수가 각 세그먼트를 순회하면서
      해당 구간의 길이와 단어 수를 계산하고 발화 속도를 도출합니다.
    """
    try:
        # 대본에서 "segments" 목록을 가져옵니다.
        segments = transcription.get("segments", [])
        
        total_time = 0.0    # 모든 세그먼트의 발화 시간을 누적합니다.
        total_words = 0     # 모든 세그먼트의 전체 단어 수.
        segment_results = []  # 각 세그먼트별 WPM 세부 정보를 저장할 리스트입니다.

        # Whisper 대본의 각 세그먼트를 직접 처리합니다.
        for seg in segments:
            # 세그먼트의 시작과 종료 시간을 가져옵니다 (초 단위)
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            duration = end - start

            # 길이가 0 이하인 세그먼트는 건너뜁니다.
            if duration <= 0:
                continue

            # 해당 세그먼트의 대본 텍스트를 가져와서 단어로 분리합니다.
            text = seg.get("text", "").strip()
            words_list = text.split()
            num_words = len(words_list)

            # 총 발화 시간과 단어 수를 누적합니다.
            total_time += duration
            total_words += num_words

            # 세그먼트의 발화 속도(WPM)을 계산합니다.
            seg_wpm = (num_words / duration) * 60 if duration > 0 else 0

            # 계산된 값을 소수점 두 자리로 반올림하여 저장합니다.
            segment_results.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "wpm": round(seg_wpm, 2),
                "words": " ".join(words_list)
            })

        # 전체 세그먼트에 대한 전체 발화 속도를 계산합니다.
        overall_wpm = (total_words / total_time) * 60 if total_time > 0 else 0

        # 최종 결과를 기존 반환 형식에 맞게 구성합니다.
        results = {
            "overall_wpm": round(overall_wpm, 2),
            "total_spoken_time": round(total_time, 2),
            "segment_wpm": segment_results
        }
        
        logging.info(f"업데이트된 발화 속도 분석 결과 -> {results}")
        return results
    
    except Exception as e:
        logging.error(f"말하기 속도 분석 중 오류 발생: {e}")
        raise


# Step 4: Volume Analysis
# 음량 분석
def analyze_volume(transcription, audio_file_path):
    """
    오디오 파일의 각 구간별 음량(RMS 및 데시벨)을 Whisper STT 반환 데이터의 세그먼트와 
    기존 오디오 파일을 이용하여 분석하는 함수입니다.
    
    매개변수:
      transcription (dict):
          Whisper가 반환한 대본으로, "segments" 배열을 포함하며 각 세그먼트는 
          "start", "end", "text" 필드를 가집니다.
      audio_file_path (str):
          원본 오디오 파일 경로.
          
    반환값:
      dict: 다음 키들을 포함하는 결과 딕셔너리:
          - "segment_data": 각 구간별 분석 결과 리스트, 각 항목은
                "time_stamps": (시작, 종료) (초 단위),
                "rms": 구간 RMS (소수점 둘째 자리 반올림),
                "db": 구간 데시벨 값 (소수점 둘째 자리 반올림)
          - "mean_rms": 모든 구간에 대한 평균 RMS (소수점 둘째 자리 반올림)
          - "mean_db": 평균 RMS를 바탕으로 계산한 평균 데시벨 (소수점 둘째 자리 반올림)
    """
    try:
        # 오디오 파일을 불러옵니다.
        audio = AudioSegment.from_file(audio_file_path)
        logging.info(f"오디오 로드 완료: {audio_file_path}")
        
        # Whisper 대본에서 세그먼트 정보를 가져옵니다.
        segments = transcription.get("segments", [])
        if not segments:
            raise ValueError("STT 반환 데이터에 'segments' 정보가 없습니다.")
        
        segment_data = []  # 각 구간별 결과를 저장할 리스트
        rms_values = []    # 각 구간의 RMS 값들을 저장
        
        # 각 세그먼트에 대해 오디오 구간을 추출하고 음량(RMS, 데시벨) 계산
        for seg in segments:
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            duration = end - start
            
            # 구간 길이가 양수인 경우에만 계산
            if duration <= 0:
                continue
            
            # pydub는 단위가 밀리초이므로 변환
            start_ms = int(start * 1000)
            end_ms = int(end * 1000)
            
            segment_audio = audio[start_ms:end_ms]
            raw_data = np.array(segment_audio.get_array_of_samples())
            
            # 해당 구간에 오디오 데이터가 없는 경우 건너뜀
            if len(raw_data) == 0:
                continue
            
            # RMS 계산 및 데시벨 변환
            rms = math.sqrt(np.mean(raw_data.astype(np.float64) ** 2))
            db = 20 * math.log10(rms + 1e-10)
            
            # 구간별 결과 저장 (소수점 둘째 자리 반올림)
            segment_data.append({
                "time_stamps": (round(start, 2), round(end, 2)),
                "rms": round(rms, 2),
                "db": round(db, 2)
            })
            rms_values.append(rms)
        
        # 전체 평균 계산
        if rms_values:
            mean_rms = np.mean(rms_values)
            mean_db = 20 * math.log10(mean_rms + 1e-10)
        else:
            mean_rms = 0
            mean_db = -math.inf
        
        result = {
            "segment_data": segment_data,
            "mean_rms": round(mean_rms, 2),
            "mean_db": round(mean_db, 2)
        }
        
        logging.info(f"음량 분석 결과 -> {result}")
        return result
    
    except Exception as e:
        logging.error(f"음량 분석 중 오류 발생: {e}")
        raise

# Example Usage
# 예제 실행
if __name__ == "__main__":
    print("테스트 용")