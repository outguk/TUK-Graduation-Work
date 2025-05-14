#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ─────────────────────────────────────────────────────────
#  FastAPI 메인 서버  (Tuk-script 통합 버전, 2025-05-15)
#    · mp4 영상 업로드  → 음성·비언어 분석
#    · txt 대본 업로드 → run_script_feedback() 으로 대본 분석
# ─────────────────────────────────────────────────────────

from __future__ import annotations

import os
import re
import json
import jwt
import uvicorn
import shutil
import asyncio
import logging
from datetime import datetime
from typing import Optional, AsyncGenerator

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Depends,
    Query,
    Header,
    Form,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

from konlpy.tag import Okt, Kkma

# ── 내부 모듈
from models.vocalization.vocalization_analysis import (
    extract_audio,
    transcribe_audio,
    analyze_speaking_speed,
    analyze_volume,
)
from models.vocalization.vocalization_evaluate import (
    evaluate_speaking_speed,
    evaluate_volume,
)
from models.nonvarbal.nonvarbal_analysis import video_nonverbal_analysis
from models.vocalization.util_functions import (
    change_sampling_rate,
    remove_noise,
    save_filtered_audio,
)
from models.script.script_feedback_korcen import run_script_feedback  # ★ 새 통합 함수

from mongodb import collection

# ─────────────────────────────────────────
# FastAPI 기본 설정
# ─────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploaded_videos")
os.makedirs(UPLOAD_DIR, exist_ok=True)

SECRET = (
    "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6"  # 데모용 시크릿 키
)
logging.info(f"FastAPI SECRET: {SECRET}")

# ─────────────────────────────────────────
# JWT 검증 유틸
# ─────────────────────────────────────────
def verify_jwt_and_get_user_id(auth_header: str) -> int:
    token = auth_header.replace("Bearer ", "")
    try:
        payload = jwt.decode(token, SECRET, algorithms=["HS384"])
        return int(payload["sub"])
    except Exception as e:
        logging.error(f"JWT decode error: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")


async def get_current_user(
    request: Request, authorization: Optional[str] = Header(default=None)
):
    if request.method == "OPTIONS":
        return None
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    return verify_jwt_and_get_user_id(authorization)


# ─────────────────────────────────────────
#  공통: 임시 파일 정리
# ─────────────────────────────────────────
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


# ─────────────────────────────────────────
# ① 영상(MP4) 업로드  →  음성·비언어 분석
# ─────────────────────────────────────────
@app.post("/fastapi/api/upload-video/")
async def upload_video(
    file: UploadFile = File(...),
    user_id: int = Depends(get_current_user),
):
    if not file.filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only MP4 files are allowed")

    # 1) 파일 저장
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    logging.info(f"Video uploaded: {file_path}")

    # 2) 오디오 추출 및 전처리
    audio_path = file_path.replace(".mp4", ".wav")
    await extract_audio(file_path, audio_path)

    temp_resampled = audio_path.replace(".wav", "_resampled.wav")
    temp_denoised = audio_path.replace(".wav", "_denoised.wav")
    temp_filtered = audio_path.replace(".wav", "_filtered.wav")

    try:
        change_sampling_rate(audio_path, 16000, temp_resampled)
        remove_noise(temp_resampled, temp_denoised)
        save_filtered_audio(temp_denoised, temp_filtered)
        os.replace(temp_filtered, audio_path)  # 최종 파일
    finally:
        for tmp in [temp_resampled, temp_denoised, temp_filtered]:
            if os.path.exists(tmp):
                os.remove(tmp)

    # 3) Whisper → 자막
    transcription = await transcribe_audio(audio_path)

    # 4) 음성·볼륨 분석 (비동기 병렬)
    loop = asyncio.get_running_loop()
    speed_task = loop.run_in_executor(None, analyze_speaking_speed, transcription)
    volume_task = loop.run_in_executor(None, analyze_volume, transcription, audio_path)
    speaking_speed, volume_analysis = await asyncio.gather(speed_task, volume_task)

    speed_score = evaluate_speaking_speed(speaking_speed)
    volume_score = evaluate_volume(volume_analysis)

    # 5) 비언어 분석
    nonverbal_analysis = video_nonverbal_analysis(os.path.abspath(file_path))

    # 6) MongoDB 저장 (대본 분석은 별도 엔드포인트에서 넣음)
    document = {
        "user_id": user_id,
        "filename": file.filename,
        "speaking_speed": speaking_speed,
        "speaking_evaluation": speed_score,
        "volume_analysis": volume_analysis,
        "volume_evaluation": volume_score,
        "nonverbal_analysis": nonverbal_analysis,
        "timestamp": datetime.utcnow().isoformat(),
    }
    result = await collection.insert_one(document)
    document["_id"] = str(result.inserted_id)
    logging.info(f"Mongo 저장 완료: {document['_id']}")

    cleanup_intermediate_files(os.path.splitext(file.filename)[0])
    return document


# ─────────────────────────────────────────
# ② 대본(txt) 업로드  →  run_script_feedback 분석
# ─────────────────────────────────────────
@app.post("/fastapi/api/analyze-script/")
async def analyze_script_endpoint(
    file: UploadFile = File(...),
    filename: str = Form(...),
    speech_minutes: int = Form(...),
    authorization: str = Header(...),
):
    user_id = verify_jwt_and_get_user_id(authorization)

    # 1) txt 파일 저장
    txt_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(txt_path, "wb") as f:
        f.write(await file.read())

    # 2) 대본 분석 호출
    analysis = run_script_feedback(
        script_path=txt_path,
        speech_minutes=speech_minutes,
        custom_badwords_path=os.path.join(BASE_DIR, "custom_profanities.txt"),
    )

    # 3) MongoDB 업데이트 (upsert)
    res = await collection.update_one(
        {"user_id": user_id, "filename": filename},
        {"$set": {"script_analysis": analysis}},
        upsert=True,
    )

    if res.matched_count == 0 and res.upserted_id:
        logging.info(f"새 문서 upsert: {res.upserted_id}")

    doc = await collection.find_one({"user_id": user_id, "filename": filename})
    doc["_id"] = str(doc["_id"])
    return JSONResponse(content=doc)


# ─────────────────────────────────────────
# ③ 분석 결과 조회
# ─────────────────────────────────────────
@app.get("/fastapi/api/get-analysis/")
async def get_analysis(
    filename: str = Query(...),
    user_id: int = Depends(get_current_user),
):
    doc = await collection.find_one({"filename": filename, "user_id": user_id})
    if not doc:
        raise HTTPException(status_code=404, detail="해당 파일 분석 결과 없음")
    doc["_id"] = str(doc["_id"])
    return doc


# ─────────────────────────────────────────
# ④ 사용자별 전체 리스트
# ─────────────────────────────────────────
@app.get("/fastapi/api/analysis-by-user/")
async def get_analysis_by_user(user_id: int = Depends(get_current_user)):
    cursor = collection.find({"user_id": user_id})
    results = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        results.append(doc)
    return {"analyses": results}


# ─────────────────────────────────────────
# ⑤ 비디오 스트리밍
# ─────────────────────────────────────────
@app.get("/fastapi/api/video/{filename}")
async def stream_video(filename: str):
    path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path, media_type="video/mp4", headers={"Accept-Ranges": "bytes"})


# ─────────────────────────────────────────
# ⑥ 대시보드 통계
# ─────────────────────────────────────────
@app.get("/fastapi/api/analysis/stats")
async def get_analysis_stats(user_id: int = Depends(get_current_user)):
    total = await collection.count_documents({"user_id": user_id})
    recent = (
        await collection.find({"user_id": user_id})
        .sort("timestamp", -1)
        .limit(5)
        .to_list(None)
    )
    recent_sanitized = [
        {"date": doc.get("timestamp", ""), "description": doc.get("filename", "")}
        for doc in recent
    ]
    return {"totalPresentations": total, "recentActivity": recent_sanitized}


# ─────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
