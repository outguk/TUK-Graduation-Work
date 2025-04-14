React SPA 라우팅 충돌 수정 (2025‑04‑14)

문제 : GET /spring/api/\*\* 요청이 항상 200 OK + index.html 로 응답되어

백엔드 컨트롤러(UserController)까지 도달하지 못함

원인 : SpringConfig#spaRouter() RouterFunction

.andRoute(RequestPredicates.GET("/\*\*") ==> … index.html …)
 → WebFlux 우선순위 때문에 모든 GET 을 가로채서 SPA fallback 수행

영향 : API 중 GET 메서드(예: /spring/api/user/profile)는

로그도 없이 HTML 반환, 프런트/포스트맨 모두 동작 실패

해결 : RouterFunction 에서 API 프리픽스 제외 (spring/api/\*\*)

테스트 방법 :

1. ./gradlew clean bootRun
2. Postman → GET http://localhost:8080/spring/api/user/profile
      Authorization: Bearer <JWT>
3. JSON 응답 & 서버 로그 확인

또는 실제 웹페이지에 접속해서 로그인 후 유저페이지 확인

<수정 파일>
GW-backend/src/main/java/TUK_Graduation_Work/GW_backend/SpringConfig.java
GW-backend/src/main/java/TUK_Graduation_Work/GW_backend/config/JwtAuthenticationFilter.java
GW-backend/src/main/java/TUK_Graduation_Work/GW_backend/controller/UserController.java
