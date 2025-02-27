현재 기존 nonvarbel 폴더의 main.py 코드를 nonvarbal_analysis.py로 옮긴 후 함수로 만들어 python-AI에서 import하여 사용해 비언어 분석을 수행하도록 수정되어있음

**통합 버전2 사용 방법**
**1.** GW_backend/src/main/java/TUK-Graduation-Work/GW-backend의 BackendApplication.java를 실행하면 톰캣 서버가 생성됨

**2.** 실행 디렉토리 python-AI에서 main.py를 실행(명령어 -> **uvicorn main:app -reload --host 0.0.0.0 --port 5000**)
   ** 2-1.** 현재 python-Ai의 main.py에서 비디오 분석 실행 부분에서 nonverbel_analysis_result로 비언어 분석 함수를 불러와 분석 결과값을 받도록 하고 이를 main.py에서 발성 분석과 같이 반환하여 엔드 서버로 보내줌

- 통합 시 중요한 것은 python-AI의 main.py함수에서 @app.post("/upload-video/") 부분을 보면 웹에서 업로드된 영상을 file_path 경로에 저장하고 있으며 비디오 분석 실행 부분에서 이 경로(file_path)를 기준으로 분석을 실행하도록 해야 함.
- 분석 시 생성되는 영상의 프레임, 키포인트나 visuailization의 경로는 바꿀 필요는 없을듯
