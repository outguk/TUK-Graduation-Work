**수정 사항**

- 현재 기존 nonvarbel 폴더의 main.py 코드를 nonvarbal_analysis.py로 옮긴 후 함수로 만들어 python-AI의 main.py에서 import하여 사용해 비언어 분석을 수행하도록 수정되어있음
- 또 python-Ai의 main.py에서 비디오 분석 실행 부분에서 nonverbel_analysis_result로 비언어 분석 함수를 불러와 분석 결과값을 받도록 하고 이를 발성 분석과 같이 반환하여 백엔드 서버로 보내주는 구조

**통합 버전2 사용 방법**

**1.** GW_backend/src/main/java/TUK-Graduation-Work/GW-backend의 BackendApplication.java를 실행하면 톰캣 서버가 생성됨(bulid.gradle의 java 버전을 본인 컴퓨터 java 버전에 맞춰주어야 함 17 or 23)

**2.** python-AI 디렉토리로 이동 해 main.py를 실행(명령어 -> **uvicorn main:app -reload --host 0.0.0.0 --port 5000**)

**3.** 이후 localhost:8080에 접속, 업로드만 테스트 시 바로 localhost:8080/upload로 들어가면 됨

- intellij 에서 BackendApplication.java를 실행하고 VSC에서 main.py를 실행하는 방식으로 진행함. VSC에서 동시에 되는 지는 해보지 않음

- 통합 시 중요한 것은 python-AI의 main.py함수에서 @app.post("/upload-video/") 부분을 보면 웹에서 업로드된 영상을 file_path 경로에 저장하고 있으며 비디오 분석 실행 부분에서 이 경로(file_path)를 기준으로 분석을 실행하도록 해야 함.
- 분석 시 생성되는 영상의 프레임, 키포인트나 visuailization의 경로는 굳이 바꿀 필요 없을듯

3/23 수정 내용

1. models/vocalization/vocalization_evaluate.py -> 음량, 속도 평가 및 점수화 코드 추가(설정한 기준에 따라 음량이 어느정도 크고 작은지, 속도가 어느정도 빠르고 느린지를 dict에 추가 + 각 구간 별 점수와 종합 점수가 측정되도록 수정)
2. main.py 에 평가 코드를 import 하여 측정 후 결과에 넘겨주도록 수정
3. 백엔드의 FastAPIcontroller에서 평가한 내용을 upload.html에 넘겨주고 화면에 평가 내용이 출력되도록 수정
