#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ─────────────────────────────────────────────────────────
#  FastAPI 메인 서버  (Tuk-script 통합 버전, 2025-05-15)
#    · mp4 영상 업로드  → 음성·비언어 분석
#    · txt 대본 업로드 → run_script_feedback() 으로 대본 분석
# ─────────────────────────────────────────────────────────

# ─────────────────────────────────────────
# FastAPI Progress 관리 추가 (2025-06-19)
#  · 영상 업로드 시 진행률 표시
# ─────────────────────────────────────────
from __future__ import annotations

import os
import re
import json
import jwt
import uvicorn
import shutil
import asyncio
import logging
import uuid
from fastapi import BackgroundTasks
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
    BackgroundTasks,    

)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

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

from mongodb import collection, script_collection
from bson import ObjectId

# ─────────────────────────────────────────
# FastAPI 기본 설정
# ─────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8080", "http://14.36.21.67:32312", "http://14.36.21.67:32313", "http://14.36.21.67:32314", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length", "Content-Type", "Access-Control-Allow-Origin"],
    
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


# progress_dict: task_id → { "stage": str, "progress": int } // 진행률 관리용
# 진행률 관리용 딕셔너리
progress_dict: dict[str, dict[str, object]] = {}

STAGES = [
    "파일 업로드 중",
    "오디오 추출 중",
    "전처리(샘플링·노이즈 제거) 중",
    "자막 변환(Whisper) 중",
    "음성·볼륨 분석 중",
    "비언어 분석 중",
    "결과 저장 및 정리 중"
]

# ─────────────────────────────────────────
# Pydantic 모델 정의 (신규 추가)
# ─────────────────────────────────────────
class ScriptRequest(BaseModel):
    script_text: str
    speech_minutes: int

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


async def process_video(filename: str, user_id: int, task_id: str):
    total = len(STAGES)
    loop = asyncio.get_running_loop() # 현재 이벤트 루프 가져오기 - 진행률 작업하면서 추가함

    try:
        for idx, stage in enumerate(STAGES[1:], start=1):
            progress_dict[task_id]["stage"] = stage
            progress_dict[task_id]["progress"] = int(idx / total * 100)
            logging.info(f"[{task_id}] {stage} ({progress_dict[task_id]['progress']}%)")

            await asyncio.sleep(0)

            # 실제 분석 단계별 작업 (예시)
            if stage == "오디오 추출 중":
                await extract_audio(
                    os.path.join(UPLOAD_DIR, filename),
                    os.path.join(UPLOAD_DIR, filename.replace(".mp4", ".wav"))
                )
            elif stage == "전처리(샘플링·노이즈 제거) 중":
                audio_path = os.path.join(UPLOAD_DIR, filename.replace(".mp4", ".wav"))
                tmp1 = audio_path.replace(".wav", "_resampled.wav")
                tmp2 = audio_path.replace(".wav", "_denoised.wav")
                tmp3 = audio_path.replace(".wav", "_filtered.wav")
                change_sampling_rate(audio_path, 16000, tmp1)
                remove_noise(tmp1, tmp2)
                save_filtered_audio(tmp2, tmp3)
                os.replace(tmp3, audio_path)
                for t in [tmp1, tmp2]:
                    if os.path.exists(t): os.remove(t)
            elif stage == "자막 변환(Whisper) 중":
                transcription = await transcribe_audio(
                    os.path.join(UPLOAD_DIR, filename.replace(".mp4", ".wav"))
                )
            elif stage == "음성·볼륨 분석 중":
                speed_task = asyncio.get_running_loop().run_in_executor(
                    None, analyze_speaking_speed, transcription
                )
                volume_task = asyncio.get_running_loop().run_in_executor(
                    None, analyze_volume, transcription,
                    os.path.join(UPLOAD_DIR, filename.replace(".mp4", ".wav"))
                )
                speaking_speed, volume_analysis = await asyncio.gather(speed_task, volume_task)
                speed_score = evaluate_speaking_speed(speaking_speed)
                volume_score = evaluate_volume(volume_analysis)
            elif stage == "비언어 분석 중":
                nonverb_task = loop.run_in_executor(
                    None,
                    video_nonverbal_analysis,
                    os.path.abspath(os.path.join(UPLOAD_DIR, filename))
                )
                nonverbal = await nonverb_task
            elif stage == "결과 저장 및 정리 중":
                doc = {
                    "user_id": user_id,
                    "filename": filename,
                    "speaking_speed": speaking_speed,
                    "speaking_evaluation": speed_score,
                    "volume_analysis": volume_analysis,
                    "volume_evaluation": volume_score,
                    "nonverbal_analysis": nonverbal,
                    "timestamp": datetime.utcnow().isoformat(),
                }
                result = await collection.insert_one(doc)
                doc["_id"] = str(result.inserted_id)
                logging.info(f"Mongo 저장 완료: {doc['_id']}")
                # 임시 파일 정리
                base = os.path.splitext(filename)[0]
                paths = [
                    os.path.join(BASE_DIR, "models", "nonvarbal", "data", "frames", base),
                    os.path.join(BASE_DIR, "models", "nonvarbal", "data", "keypoints", base),
                    os.path.join(BASE_DIR, "models", "nonvarbal", "data", "visualizations", base),
                    os.path.join(BASE_DIR, "models", "nonvarbal", "data", "test_results.pkl"),
                    os.path.join(BASE_DIR, "models", "nonvarbal", "data", "inference_results.json"),
                ]
                for p in paths:
                    if os.path.isdir(p): shutil.rmtree(p)
                    elif os.path.isfile(p): os.remove(p)
        # 완료 상태
        progress_dict[task_id]["stage"] = "완료"
        progress_dict[task_id]["progress"] = 100
        logging.info(f"[{task_id}] 분석 완료")
    except Exception as e:
        logging.error(f"[{task_id}] 분석 중 오류: {e}")
        progress_dict[task_id]["stage"] = "오류 발생"
        raise



@app.post("/fastapi/api/upload-video/", status_code=202)  #status_code 수정 (202)
async def upload_video(
    background_tasks: BackgroundTasks,   
    file: UploadFile = File(...),
    user_id: int = Depends(get_current_user),
):
    if not file.filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only MP4 files are allowed")

    # 추가: task_id 생성 및 초기화
    task_id = str(uuid.uuid4())
    progress_dict[task_id] = {"stage": STAGES[0], "progress": 0}

    # 파일 저장
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(save_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)
    logging.info(f"[{task_id}] {STAGES[0]} 완료: {save_path}")

    # 추가: 백그라운드 작업 시작
    background_tasks.add_task(process_video, file.filename, user_id, task_id)

    # 반환값 수정
    return {"task_id": task_id,      "filename": file.filename  
}

# ─────────────────────────────────────────
@app.get("/fastapi/api/progress/{task_id}")  # 진행률 엔드포인트
async def get_progress(task_id: str):
    info = progress_dict.get(task_id)
    if not info:
        raise HTTPException(status_code=404, detail="Invalid task_id")
    return info  # {"stage": "...", "progress": 30}








# # ─────────────────────────────────────────
# # ① 영상(MP4) 업로드  →  음성·비언어 분석
# # ─────────────────────────────────────────
# @app.post("/fastapi/api/upload-video/", status_code=202)
# async def upload_video(
#     file: UploadFile = File(...),
#     user_id: int = Depends(get_current_user),
# ):
#     if not file.filename.endswith(".mp4"):
#         raise HTTPException(status_code=400, detail="Only MP4 files are allowed")

#     # 1) 파일 저장
#     file_path = os.path.join(UPLOAD_DIR, file.filename)
#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
#     logging.info(f"Video uploaded: {file_path}")

#     # 2) 오디오 추출 및 전처리
#     audio_path = file_path.replace(".mp4", ".wav")
#     await extract_audio(file_path, audio_path)

#     temp_resampled = audio_path.replace(".wav", "_resampled.wav")
#     temp_denoised = audio_path.replace(".wav", "_denoised.wav")
#     temp_filtered = audio_path.replace(".wav", "_filtered.wav")

#     try:
#         change_sampling_rate(audio_path, 16000, temp_resampled)
#         remove_noise(temp_resampled, temp_denoised)
#         save_filtered_audio(temp_denoised, temp_filtered)
#         os.replace(temp_filtered, audio_path)  # 최종 파일
#     finally:
#         for tmp in [temp_resampled, temp_denoised, temp_filtered]:
#             if os.path.exists(tmp):
#                 os.remove(tmp)

#     # 3) Whisper → 자막
#     transcription = await transcribe_audio(audio_path)

#     # 4) 음성·볼륨 분석 (비동기 병렬)
#     loop = asyncio.get_running_loop()
#     speed_task = loop.run_in_executor(None, analyze_speaking_speed, transcription)
#     volume_task = loop.run_in_executor(None, analyze_volume, transcription, audio_path)
#     speaking_speed, volume_analysis = await asyncio.gather(speed_task, volume_task)

#     speed_score = evaluate_speaking_speed(speaking_speed)
#     volume_score = evaluate_volume(volume_analysis)

#     # 5) 비언어 분석
#     nonverbal_analysis = video_nonverbal_analysis(os.path.abspath(file_path))

#     # 6) MongoDB 저장 (대본 분석은 별도 엔드포인트에서 넣음)
#     document = {
#         "user_id": user_id,
#         "filename": file.filename,
#         "speaking_speed": speaking_speed,
#         "speaking_evaluation": speed_score,
#         "volume_analysis": volume_analysis,
#         "volume_evaluation": volume_score,
#         "nonverbal_analysis": nonverbal_analysis,
#         "timestamp": datetime.utcnow().isoformat(),
#     }
#     result = await collection.insert_one(document)
#     document["_id"] = str(result.inserted_id)
#     logging.info(f"Mongo 저장 완료: {document['_id']}")

#     cleanup_intermediate_files(os.path.splitext(file.filename)[0])
#     return document


# ─────────────────────────────────────────
# ② 대본(txt) 업로드  →  run_script_feedback 분석
# ─────────────────────────────────────────
@app.post("/fastapi/api/analyze-script/")
async def analyze_script_endpoint(
    file: UploadFile = File(...),
    filename: str = Form(...),
    speech_minutes: int = Form(...),
    user_id: int = Depends(get_current_user),
):
    # txt 파일 저장
    txt_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(txt_path, "wb") as f:
        f.write(await file.read())

    # 대본 분석 호출
    analysis = run_script_feedback(
        script_path=txt_path,
        speech_minutes=speech_minutes,
        custom_badwords_path=os.path.join(BASE_DIR, "custom_profanities.txt"),
    )

    # MongoDB 저장 (script_collection)
    doc = {
        "user_id": user_id,
        "script_text": open(txt_path, "r", encoding="utf-8").read(),
        "script_analysis": analysis,
        "speech_minutes": speech_minutes,
        "filename": filename,  # 영상 파일명과 연관성 유지
        "timestamp": datetime.utcnow().isoformat(),
    }
    result = await script_collection.insert_one(doc)
    doc["_id"] = str(result.inserted_id)
    logging.info(f"Script analysis saved: {doc['_id']}")

    return JSONResponse(content=doc)

# ─────────────────────────────────────────
# ③ 텍스트 입력 대본 분석 (신규 추가)
# ─────────────────────────────────────────
@app.post("/fastapi/api/analyze-text-script/")
async def analyze_text_script(
    request: ScriptRequest,
    user_id: int = Depends(get_current_user),
):
    # 대본 분석 호출
    analysis = run_script_feedback(
        script_path=None,
        script_text=request.script_text,
        speech_minutes=request.speech_minutes,
        custom_badwords_path=os.path.join(BASE_DIR, "custom_profanities.txt"),
    )

    # MongoDB 저장 (script_collection)
    doc = {
        "user_id": user_id,
        "script_text": request.script_text,
        "script_analysis": analysis,
        "speech_minutes": request.speech_minutes,
        "filename": f"text_script_{str(uuid.uuid4())}",  # 고유 파일명 생성
        "timestamp": datetime.utcnow().isoformat(),
    }
    result = await script_collection.insert_one(doc)
    doc["_id"] = str(result.inserted_id)
    logging.info(f"Script analysis saved: {doc['_id']}")

    return JSONResponse(content=doc)

# ─────────────────────────────────────────
# ④ 대본 분석 결과 조회 (신규 추가)
# ─────────────────────────────────────────
@app.get("/fastapi/api/get-script-analysis/")
async def get_script_analysis(
    script_id: str = Query(...),
    user_id: int = Depends(get_current_user),
):
    doc = await script_collection.find_one({"_id": ObjectId(script_id), "user_id": user_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Script analysis not found")
    doc["_id"] = str(doc["_id"])
    return doc

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
# ⑥ 사용자별 전체 리스트 (수정: 대본 분석 포함)
# ─────────────────────────────────────────
@app.get("/fastapi/api/analysis-by-user/")
async def get_analysis_by_user(user_id: int = Depends(get_current_user)):
    # 영상 분석 결과
    cursor = collection.find({"user_id": user_id})
    video_results = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        video_results.append(doc)
    
    # 대본 분석 결과
    script_cursor = script_collection.find({"user_id": user_id})
    script_results = []
    async for doc in script_cursor:
        doc["_id"] = str(doc["_id"])
        script_results.append(doc)
    
    return {
        "video_analyses": video_results,
        "script_analyses": script_results
    }


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
# ⑧ 대시보드 통계 (수정: 대본 분석 통계 포함)
# ─────────────────────────────────────────
@app.get("/fastapi/api/analysis/stats")
async def get_analysis_stats(user_id: int = Depends(get_current_user)):
    # 영상 분석 통계
    video_total = await collection.count_documents({"user_id": user_id})
    video_recent = (
        await collection.find({"user_id": user_id})
        .sort("timestamp", -1)
        .limit(5)
        .to_list(None)
    )
    video_recent_sanitized = [
        {"date": doc.get("timestamp", ""), "description": doc.get("filename", "")}
        for doc in video_recent
    ]
    
    # 대본 분석 통계
    script_total = await script_collection.count_documents({"user_id": user_id})
    script_recent = (
        await script_collection.find({"user_id": user_id})
        .sort("timestamp", -1)
        .limit(5)
        .to_list(None)
    )
    script_recent_sanitized = [
        {"date": doc.get("timestamp", ""), "description": doc.get("filename", "")}
        for doc in script_recent
    ]
    
    return {
        "totalVideoPresentations": video_total,
        "recentVideoActivity": video_recent_sanitized,
        "totalScriptAnalyses": script_total,
        "recentScriptActivity": script_recent_sanitized
    }


# ─────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
