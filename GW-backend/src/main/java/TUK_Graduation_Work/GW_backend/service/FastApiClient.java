package TUK_Graduation_Work.GW_backend.service;

import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Service;
import org.springframework.http.MediaType;
import org.springframework.core.io.FileSystemResource;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import java.io.File;
import java.util.Map;

@Service
public class FastApiClient {
    private final WebClient webClient = WebClient.builder().baseUrl("http://localhost:5000").build();

    /**
     * 비디오 파일을 FastAPI에 업로드합니다. JWT 토큰을 통해 사용자 인증.
     * @param filePath 로컬 파일 경로
     * @param token JWT 토큰
     * @return FastAPI 응답 (JSON 문자열)
     */
    public Mono<String> uploadFileToFastAPI(String filePath, String token) {
        File videoFile = new File(filePath);
        if (!videoFile.exists() || !videoFile.isFile()) {
            return Mono.error(new RuntimeException("Error: 업로드할 파일을 찾을 수 없습니다!"));
        }
    
        System.out.println("📂 FastAPI로 업로드 요청: " + filePath);
    
        FileSystemResource fileResource = new FileSystemResource(videoFile);
    
        return webClient.post()
                // 수정: main.py에서 /fastapi/api/upload-video 로 라우트 변경
                .uri("/fastapi/api/upload-video/")
                .header("Authorization", "Bearer " + token)
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(BodyInserters.fromMultipartData("file", fileResource))
                .retrieve()
                .onStatus(HttpStatusCode::is4xxClientError, clientResponse ->
                        Mono.error(new RuntimeException("FastAPI 요청 오류 (4xx): " + clientResponse.statusCode()))
                )
                .onStatus(HttpStatusCode::is5xxServerError, clientResponse ->
                        Mono.error(new RuntimeException("FastAPI 서버 오류 (5xx): " + clientResponse.statusCode()))
                )
                .bodyToMono(String.class)
                .doOnError(error -> System.err.println("🚨 FastAPI 요청 실패: " + error.getMessage()));
    }
    /**
     * 사용자별 분석 결과를 조회합니다. JWT 토큰으로 인증.
     * @param token JWT 토큰
     * @return FastAPI 응답 (Map 형식)
     */
    public Mono<Map<String, Object>> fetchAnalysesByUser(String token) {
        return webClient.get()
                // 수정: main.py에서 /fastapi/api/analysis-by-user/ 로 라우트 변경
                .uri("/fastapi/api/analysis-by-user/")
                .header("Authorization", "Bearer " + token)
                .retrieve()
                .onStatus(HttpStatusCode::is4xxClientError, clientResponse ->
                        Mono.error(new RuntimeException("FastAPI 요청 오류 (4xx): " + clientResponse.statusCode()))
                )
                .onStatus(HttpStatusCode::is5xxServerError, clientResponse ->
                        Mono.error(new RuntimeException("FastAPI 서버 오류 (5xx): " + clientResponse.statusCode()))
                )
                .bodyToMono(new org.springframework.core.ParameterizedTypeReference<Map<String, Object>>() {})
                .doOnError(error -> System.err.println("🚨 FastAPI 요청 실패: " + error.getMessage()));
    }
    /**
     * 분석 통계를 조회합니다. JWT 토큰으로 인증.
     * @param token JWT 토큰
     * @return FastAPI 응답 (Map 형식)
     */
    public Mono<Map<String, Object>> fetchAnalysisStats(String token) {
        return webClient.get()
                // 수정: main.py에서 /fastapi/api/analysis/stats 로 라우트 변경
                .uri("/fastapi/api/analysis/stats")
                .header("Authorization", "Bearer " + token)
                .retrieve()
                .onStatus(HttpStatusCode::is4xxClientError, clientResponse ->
                    Mono.error(new RuntimeException("FastAPI 요청 오류 (4xx): " + clientResponse.statusCode()))
                )
                .onStatus(HttpStatusCode::is5xxServerError, clientResponse ->
                    Mono.error(new RuntimeException("FastAPI 서버 오류 (5xx): " + clientResponse.statusCode()))
                )
                .bodyToMono(new org.springframework.core.ParameterizedTypeReference<Map<String, Object>>() {})
                .doOnError(error -> System.err.println("🚨 FastAPI 요청 실패: " + error.getMessage()));
    }

    public Mono<Map> getAnalysis(String filename, String authorization) {
        return webClient.get()
                .uri(uriBuilder -> uriBuilder
                        .path("/fastapi/api/get-analysis/")
                        .queryParam("filename", filename)
                        .build())
                .header("Authorization", authorization)
                .retrieve()
                .onStatus(status -> status.isError(), response -> {
                    return response.bodyToMono(String.class)
                            .flatMap(errorBody -> {
                                return Mono.error(new RuntimeException("FastAPI error: " + errorBody));
                            });
                })
                .bodyToMono(Map.class);
    }
}