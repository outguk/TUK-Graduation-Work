from typing import Optional, Union, AsyncGenerator
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query, Header, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
import asyncio
import json
import logging
import shutil
import re
from datetime import datetime
import jwt


from fastapi.responses import FileResponse

# 기존 모듈 임포트 유지
from models.vocalization.vocalization_analysis import extract_audio, transcribe_audio, analyze_speaking_speed, analyze_volume
from models.nonvarbal.nonvarbal_analysis import video_nonverbal_analysis
from models.vocalization.vocalization_evaluate import evaluate_speaking_speed, evaluate_volume
from models.script.script_feedback_korcen import load_custom_badwords, analyze_script, evaluate_length
from models.vocalization.util_functions import (
    change_sampling_rate,
    remove_noise,
    save_filtered_audio
)
from konlpy.tag import Okt, Kkma

from mongodb import collection

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_videos")
os.makedirs(UPLOAD_DIR, exist_ok=True)

SCRIPT_PATH = os.path.join(BASE_DIR, "models", "script", "script_feedback_korcen.py")
def verify_jwt_and_get_user_id(auth_header: str) -> int:
    token = auth_header.replace("Bearer ", "")
    payload = jwt.decode(token, SECRET, algorithms=["HS384"])
    return int(payload["sub"])

SECRET = "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6"

def verify_jwt_and_get_user_id(auth_header: str) -> int:  # ★ JWT → user_id 추출
    token = auth_header.replace("Bearer ", "")
    try:
        payload = jwt.decode(token, SECRET, algorithms=["HS384"])
        return int(payload["sub"])
    except Exception as e:
        logging.error(f"JWT decode error: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")


logging.info(f"FastAPI SECRET: {SECRET}")

# 토큰 인증 의존성
async def get_current_user(request: Request, authorization: Optional[str] = Header(default=None)):
    if request.method == "OPTIONS":
        return None
    if not authorization:
        logging.info("No Authorization header provided")
        raise HTTPException(status_code=401, detail="Authorization header missing")
    try:
        token = authorization.replace("Bearer ", "")
        logging.info(f"Extracted token: {token}")
        decoded_header = jwt.get_unverified_header(token)
        decoded_payload = jwt.decode(token, options={"verify_signature": False})
        logging.info(f"Token header: {decoded_header}")
        logging.info(f"Token payload: {decoded_payload}")
        
        # HS384 허용
        payload = jwt.decode(token, SECRET, algorithms=["HS384"])
        user_id = payload.get("sub")
        if user_id is None:
            logging.error("Invalid token: user_id not found in sub")
            raise HTTPException(status_code=401, detail="Invalid token: user_id not found in sub")
        logging.info(f"Validated token, user_id: {user_id}")
        return int(user_id)
    except jwt.ExpiredSignatureError:
        logging.error("Token has expired")
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError as e:
        logging.error(f"Invalid token error: {str(e)}")
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error during token validation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Token validation failed: {str(e)}")

# 나머지 함수 및 엔드포인트는 그대로 유지
def cleanup_intermediate_files(video_name: str):
    data_dir = os.path.join(BASE_DIR, "models", "nonvarbal", "data")
    paths = [
        os.path.join(data_dir, "frames", video_name),
        os.path.join(data_dir, "keypoints", video_name),
        os.path.join(data_dir, "visualizations", video_name),
        os.path.join(data_dir, "test_results.pkl"),
        os.path.join(data_dir, "inference_results.json"),
    ]
    for path in paths[:3]:
        if os.path.isdir(path):
            shutil.rmtree(path)
            logging.info(f"[CLEANUP] 디렉토리 삭제: {path}")
    for path in paths[3:]:
        if os.path.isfile(path):
            os.remove(path)
            logging.info(f"[CLEANUP] 파일 삭제: {path}")
    logging.info(f"[CLEANUP] {video_name} 분석 완료.")

@app.post("/fastapi/api/upload-video/")
async def upload_video(file: UploadFile = File(...), user_id: int = Depends(get_current_user)):
    if not file.filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only MP4 files are allowed")
    # 나머지 로직 동일
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    logging.info(f"Video uploaded: {file_path}")
    audio_path = file_path.replace(".mp4", ".wav")
    await extract_audio(file_path, audio_path)

    # 전처리 임시 경로 설정
    temp_resampled = audio_path.replace(".wav", "_resampled.wav")
    temp_denoised = audio_path.replace(".wav", "_denoised.wav")
    temp_filtered = audio_path.replace(".wav", "_filtered.wav")

    try:
        # 1단계: 샘플링 레이트 16kHz로 변경
        change_sampling_rate(audio_path, 16000, temp_resampled)

        # 2단계: 노이즈 제거
        remove_noise(temp_resampled, temp_denoised)

        # 3단계: 대역 통과 필터 적용
        save_filtered_audio(temp_denoised, temp_filtered)

        # 최종 파일 audio_path로 덮어쓰기 (temp_filtered 파일을 이동)
        os.replace(temp_filtered, audio_path)
    except Exception as e:
        logging.error("오디오 전처리 중 오류 발생: %s", e)
        raise
    finally:
        # 임시 파일 삭제 (최종 파일(audio_path)과 경로가 동일하지 않은 경우에만 삭제)
        for temp_path in [temp_resampled, temp_denoised, temp_filtered]:
            if os.path.abspath(temp_path) != os.path.abspath(audio_path) and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    logging.info(f"🧹 임시 파일 삭제됨: {temp_path}")
                except Exception as e:
                    logging.warning(f"⚠️ 임시 파일 삭제 실패: {temp_path} | {e}")


    transcription = await transcribe_audio(audio_path)

    script_txt_path = os.path.join(BASE_DIR, "output_transcription.txt")
    if not os.path.exists(script_txt_path):
        raise HTTPException(status_code=500, detail="Transcription file not found")
    with open(script_txt_path, "rb") as f:
        text = f.read().decode('utf-8', errors='ignore')
    sentences = [s.strip() for s in re.split(r"[.!?\n]", text) if s.strip()]
    tagger = Okt()
    kkma = Kkma()
    badwords_path = os.path.join(BASE_DIR, "custom_profanities.txt")
    custom_badwords = load_custom_badwords(badwords_path)
    script_stats = analyze_script(sentences, tagger, kkma, custom_badwords)
    actual_chars, min_chars, max_chars, length_feedback = evaluate_length(text, speech_minutes=3)

    loop = asyncio.get_running_loop()
    speed_task = loop.run_in_executor(None, analyze_speaking_speed, transcription)
    volume_task = loop.run_in_executor(None, analyze_volume, transcription, audio_path)
    speaking_speed, volume_analysis = await asyncio.gather(speed_task, volume_task)

    absolute_file_path = os.path.abspath(file_path)
    nonverbal_analysis = video_nonverbal_analysis(absolute_file_path)

    speed_score = evaluate_speaking_speed(speaking_speed)
    volume_score = evaluate_volume(volume_analysis)
    
    document = {
        "user_id": user_id,
        "filename": file.filename,
        "speaking_speed": speaking_speed,
        "speaking_evaluation": speed_score,
        "volume_analysis": volume_analysis,
        "volume_evaluation": volume_score,
        "nonverbal_analysis": nonverbal_analysis,
        "script_analysis": {
            "length": actual_chars,
            "length_feedback": length_feedback,
            "min_length": min_chars,
            "max_length": max_chars,
            "non_honorific_count": script_stats["non_honorific_count"],
            "uncertainty_count": script_stats["uncertainty_count"],
            "subject_verb_mismatch_count": script_stats["subject_verb_mismatch_count"],
            "profanity_count": script_stats["profanity_count"],
            "non_honorific_examples": script_stats["non_honorific_examples"],
            "uncertainty_examples": script_stats["uncertainty_examples"],
            "subject_verb_examples": script_stats["subject_verb_examples"],
            "profanity_examples": script_stats["profanity_examples"],
            "otas_detected": script_stats["otas_detected"],
        },
        "timestamp": datetime.utcnow().isoformat(),
    }
    inserted = await collection.insert_one(document)
    logging.info(f"분석 결과 MongoDB에 저장함: ID={inserted.inserted_id}")
    video_name = os.path.splitext(file.filename)[0]
    cleanup_intermediate_files(video_name)
    document["_id"] = str(inserted.inserted_id)
    return document

@app.get("/fastapi/api/get-analysis/")
async def get_analysis(filename: str = Query(..., description="업로드된 파일 이름"), user_id: int = Depends(get_current_user)):
    document = await collection.find_one({"filename": filename, "user_id": user_id})
    if not document:
        raise HTTPException(status_code=404, detail="해당 파일 분석 결과 없음")
    document["_id"] = str(document["_id"])
    return document

@app.get("/fastapi/api/analysis-by-user/")
async def get_analysis_by_user(user_id: int = Depends(get_current_user)):
    cursor = collection.find({"user_id": user_id})
    results = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        results.append(doc)
    return {"analyses": results}

@app.get("/fastapi/api/video/{filename}")
async def stream_video(filename: str):
    file_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        headers={"Accept-Ranges": "bytes"}
    )


@app.get("/fastapi/api/analysis/stats")
async def get_analysis_stats(user_id: int = Depends(get_current_user)):
    try:
        # 총 발표 수
        total_presentations = await collection.count_documents({"user_id": user_id})
        logging.info(f"Total presentations for user_id {user_id}: {total_presentations}")

        # 최근 활동 (최대 5개)
        recent_activity = await collection.find({"user_id": user_id}).sort("timestamp", -1).limit(5).to_list(None)

        # 안전한 데이터 처리
        sanitized_activity = []
        for activity in recent_activity:
            try:
                sanitized_activity.append({
                    "date": activity.get("timestamp", ""),  # timestamp 없으면 빈 문자열
                    "description": activity.get("filename", "Unknown")  # filename 없으면 기본값
                })
            except Exception as e:
                logging.error(f"Error processing activity for user_id {user_id}: {e}, activity: {activity}")
                continue

        response = {
            "totalPresentations": total_presentations,
            "recentActivity": sanitized_activity
        }
        logging.info(f"Analysis stats response for user_id {user_id}: {response}")
        return response
    except Exception as e:
        logging.error(f"Error in get_analysis_stats for user_id {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch analysis stats: {str(e)}")

@app.post("/fastapi/api/analyze-script/")
async def analyze_script_endpoint(
    file: UploadFile = File(...),
    filename: str = Form(...),
    authorization: str = Header(...)
):
    user_id = verify_jwt_and_get_user_id(authorization)

    # 1) 임시 저장
    tmp_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    # 2) 외부 스크립트 호출
    result = subprocess.run(
      ["python3", SCRIPT_PATH, tmp_path],
      capture_output=True, text=True
    )
    if result.returncode != 0:
      raise HTTPException(500, "Script error: "+result.stderr)

    analysis = json.loads(result.stdout)

    # 3) MongoDB 업데이트
    res = collection.update_one(
      {"user_id": user_id, "filename": filename},
      {"$set": {"script_analysis": analysis}}
    )
    if res.matched_count == 0:
      raise HTTPException(404, "Presentation not found")

    # 4) 업데이트된 문서 조회 및 반환
    doc = collection.find_one({"user_id": user_id, "filename": filename})
    return JSONResponse(content=doc)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)