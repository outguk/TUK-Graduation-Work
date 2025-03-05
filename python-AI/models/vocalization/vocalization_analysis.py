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
from pydub import AudioSegment, silence
from collections import Counter
from cmudict import entries as cmu_dict

# 프로젝트 루트를 sys.path에 추가(경로 검색을 쉽게 하기 위해)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# 디렉토리 내 필요 모듈 import
from models.vocalization.util_functions import process_audio_chunks, change_sampling_rate

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
    await asyncio.sleep(1)
    # 오디오 추출 시작 메시지 출력
    print(f"Extracting audio from video: {video_file}")
    # MoviePy를 사용하여 동영상 파일을 로드하고 오디오 트랙 추출
    try:
      video = mp.VideoFileClip(video_file)
    except FileNotFoundError:
      print(f"Error: Video file '{video_file}' not found.")
      return None

    video.audio.write_audiofile(output_audio_file)
    # 오디오 추출 완료 후 메시지 출력
    print(f"Audio extracted to: {output_audio_file}")
    return output_audio_file

# Step 2: Speech-to-Text with Whisper
# 음성을 텍스트로 변환 (Whisper 사용) -> 최적화 및 개선 필요
async def transcribe_audio(audio_file, model_name="base"):
    await asyncio.sleep(1)
    # 텍스트 변환 시작 메시지 출력
    print(f"Transcribing audio: {audio_file} with model: {model_name}")
    # Whisper 모델 로드
    model = whisper.load_model(model_name)
    # 로드된 모델로 음성 텍스트 변환 수행
    transcription = model.transcribe(audio_file) # dict 형태로 반환

    ''' 출력된 대본을 파일 형태로 저장할 필요가 있음 (test 코드처럼) '''

    # 변환된 텍스트의 첫 100자를 출력하여 검증
    # print(f"Transcription completed. Text: {transcription['text'][:100]}...")
    logging.info(transcription['text']) 
    return transcription

# Step 3: Analyze Speaking Speed (Words per Minute)
# 말하기 속도 분석 (분당 단어 수 계산) -> 최적화 및 개선 필요(무음 구간 추출 부분 모듈화화)
def analyze_speaking_speed(transcription, audio_file_path, min_silence_len=1000, silence_thresh=-40):
    """
    util의 음성 구간 추정 함수를 활용해 말하기 속도를 계산하는 함수.

    Returns:
        dict: 전체 WPM, 발화 구간별 WPM, 총 발화 시간.
    """

    # Step 1: 무음 무간 추정 함수를 통해 발화 구간 불러오기기
    audio, non_silent_chunks = process_audio_chunks(audio_file_path, min_silence_len, silence_thresh)

    # Step 2: 총 발화 시간 계산
    total_spoken_time = sum(end - start for start, end in non_silent_chunks)

    # Step 3: 말하기 속도 계산
    words = transcription['text'].split() # 스크립트의 단어 분리
    num_words = len(words) # 전체 단어 수
    overall_wpm = (num_words / total_spoken_time) * 60 if total_spoken_time > 0 else 0 # 전체 발화 구간 평균 WPM

    # Step 4: 발화 구간별 분석
    segment_wpm = []
    word_index = 0
    word_time = total_spoken_time / num_words if num_words > 0 else 0

    for start, end in non_silent_chunks: # 각 발화 구간 별
        segment_duration = end - start # 구간별 발화 시간 계산
        segment_words = int(segment_duration / word_time) if word_time > 0 else 1 # 구간별 평균 단어 수(발화 시간이 짧게 측정돼 0이 되는 경우 1로 기본값 설정)
        wpm = round((segment_words / segment_duration) * 60,2) if segment_duration > 0 else 0 # WPM 계산
        # 구간별 결과 저장
        segment_wpm.append({
            "start": round(start,2),
            "end": round(end,2),
            "wpm": wpm,
            "words": " ".join(words[word_index:word_index + segment_words])
        })
        word_index += segment_words

    # Step 5: 결과 반환
    results= {
        "overall_wpm": round(overall_wpm,2),
        "total_spoken_time": round(total_spoken_time,2),
        # "segment_wpm": segment_wpm 구간별 wpm -> 나중에 구간 별 피드백 용
    }
    logging.info(f"속도 분석 결과 -> {results}")
    return results


# Step 4: Volume Analysis
# 음량 분석
def analyze_volume(audio_file_path, min_silence_len=1000, silence_thresh=-40):
    """
    오디오 파일의 음성 구간별 음량(RMS 및 데시벨)을 분석하고 시각화.

    매개변수:
        file_path (str): 오디오 파일 경로.
        min_silence_len (int): 무음으로 간주할 최소 길이 (ms).
        silence_thresh (int): 무음으로 간주할 데시벨 임계값. -> 최적이 무엇일 지 생각 필요

    반환값:
        dict: 시간 구간, RMS 값, 데시벨 값의 리스트.
    """
    try:
        # Step 1: 무음 무간 추정 함수를 통해 발화 구간 불러오기
        audio, non_silent_chunks = process_audio_chunks(audio_file_path, min_silence_len, silence_thresh)

        # Step 2. 자료 구조 정의
        '''각 음성 구간(non_silent_chunks)에 대해 음량 분석'''
        segment_data=[] # 각 시간 구간 별 정보 저장 리스트
        rms_values = [] # 각 구간의 RMS 값. (dB 변환에 필요)
        db_values = [] # 각 구간의 데시벨 값.

        # Step 3. 발화 구간 별 음량 분석
        for start, end in non_silent_chunks:
            segment = audio[int(start * 1000):int(end * 1000)]# 시작과 끝 밀리초를 사용하여 해당 구간의 오디오 데이터를 추출
            raw_data = np.array(segment.get_array_of_samples()) # 오디오 데이터를 샘플 값(PCM 데이터)로 변환하여 NumPy 배열로 가져옴
            # 빈 구간 처리 (소리가 비어있다면 분석할 필요 x)
            if len(raw_data) == 0:
                continue 
            # NaN 및 Infinity 값 처리 (Why?)
            if np.isnan(raw_data).any() or np.isinf(raw_data).any():
                logging.warning(f"NaN 또는 Infinity 값 감지: {start}ms - {end}ms 구간을 건너뜁니다.")
                continue
          
            rms = np.sqrt(np.mean(raw_data**2)) #
            db = 20 * np.log10(rms + 1e-10)

            # 각 구간 별 결과 저장
            segment_data.append({
                "time_stamps":(start,end),
                "rms":round(rms,2),
                "db":round(db,2)
            })

            # Step 4. 전체 평균 음량 계산
            if rms_values:
                mean_rms = np.mean(rms_values)  # RMS 평균
                mean_db = 20 * np.log10(mean_rms + 1e-10)  # 평균 dB 변환
            else:
                mean_rms = 0
                mean_db = -np.inf  # 평균을 구할 데이터가 없으면 -∞ dB 처리
            
            rms_values.append(round(rms,2))
            db_values.append(round(db,2))

        # # 4. 결과 시각화 (나중에 조정 필요)
        # avg_times = [(start + end) / 2 for start, end in time_stamps]

        # plt.figure(figsize=(10, 6))
        # plt.plot(avg_times, db_values, marker='o', label="Volume (dB)")
        # plt.xlabel("Time (s)")
        # plt.ylabel("Volume (dB)")
        # plt.title("Segment-wise Volume Analysis")
        # plt.grid()
        # plt.legend()
        # plt.show()

        # 5. 결과 반환(각 구간별 결과가 저장됨)
        result = {
            # 구간 별 측정 치(테스트를 위해 숨겨둠)
            # "segment_data": segment_data,
            "mean_rms": mean_rms,
            "mean_db": mean_db
        }
        logging.info(f"음량 분석 결과 -> {result}")
        return result
    
    except Exception as e:
        logging.info("예상치 못한 오류가 발생했습니다.")

# Step 5: Pronunciation Analysis
# 발음 분석
def analyze_pronunciation(transcription):
    # 발음 분석 시작 메시지 출력
    print(f"Analyzing pronunciation for transcription text.")
    # 변환된 텍스트를 소문자로 변환하고 단어로 분리
    words = transcription['text'].lower().split()
    # CMU 발음 사전 로드
    cmu = {word: phonemes for word, phonemes in cmu_dict()}
    pronunciation_issues = []

    # 각 단어를 CMU 사전과 대조
    for word in words:
        if word not in cmu:
            pronunciation_issues.append(word)  # 사전에 없는 단어를 기록

    # 발음 분석 결과 출력
    print(f"Pronunciation analysis - Total words: {len(words)}, Mispronounced: {len(pronunciation_issues)}")
    return {
        "total_words": len(words),
        "mispronounced_words": pronunciation_issues,
    }

# Main Function
# 메인 함수 (모듈화 버전, 나중에 통합)
def analyze_presentation(video_file):
    # 분석 프로세스 시작 메시지 출력
    print(f"Starting analysis for video file: {video_file}")
    # Step 1: 동영상에서 오디오 트랙 추출
    audio_file = extract_audio(video_file)

    # Step 2: 오디오 데이터 전처리
    audio_path = os.path.abspath("tests/test_video/test1_audio.wav")
    pre_audio_path = os.path.abspath("tests/test_video/test1_audio.wav")
    change_sampling_rate(audio_path, 16000, pre_audio_path)

    # Step 2: 추출된 오디오 텍스트 변환
    transcription = transcribe_audio(audio_file)

    # Pydub을 사용하여 오디오 파일 로드 후 길이 측정 (테스트 때는 안해봐서 오류 가능성)
    audio = AudioSegment.from_wav(audio_file)
    duration = len(audio) / 1000  # 밀리초를 초로 변환

    # Step 3: 말하기 속도 계산
    speaking_speed = analyze_speaking_speed(transcription, duration)
    # Step 4: 음량 분석 수행
    volume_analysis = analyze_volume(audio_file)
    # Step 5: 발음 분석 수행
    # pronunciation_analysis = analyze_pronunciation(transcription)

    # 결과를 딕셔너리로 정리
    results = {
        "speaking_speed_wpm": speaking_speed,
        "volume_analysis": volume_analysis,
    }

    # 분석 완료 메시지 출력
    print("Analysis completed. Saving results to analysis_results.json")
    # 결과를 JSON 파일로 저장
    with open("analysis_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # 결과 저장 성공 메시지 출력
    print("Results saved successfully.")
    return results

# Example Usage
# 예제 실행
if __name__ == "__main__":
    # 처리할 동영상 파일 경로 지정
    video_path = "presentation.mp4"  # 동영상 파일 경로로 변경
    print(f"Processing video: {video_path}")
    # 분석 수행 및 최종 결과 출력
    results = analyze_presentation(video_path)
    print("Final Results:")
    print(json.dumps(results, indent=4))