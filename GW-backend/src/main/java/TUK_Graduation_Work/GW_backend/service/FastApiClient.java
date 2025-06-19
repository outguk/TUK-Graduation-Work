package TUK_Graduation_Work.GW_backend.service;

import java.io.File;
import java.util.Map;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.core.io.FileSystemResource;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;

import reactor.core.publisher.Mono;

@Service
public class FastApiClient {
    private final WebClient webClient = WebClient.builder().baseUrl("http://localhost:5000").build();
    private static final Logger logger = LoggerFactory.getLogger(FastApiClient.class);

    public Mono<String> uploadFileToFastAPI(String filePath, String token) {
        File videoFile = new File(filePath);
        if (!videoFile.exists() || !videoFile.isFile()) {
            return Mono.error(new RuntimeException("Error: 업로드할 파일을 찾을 수 없습니다!"));
        }
    
        logger.info("📂 FastAPI로 업로드 요청: " + filePath);
    
        FileSystemResource fileResource = new FileSystemResource(videoFile);
    
        return webClient.post()
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
                .doOnError(error -> logger.error("🚨 FastAPI 요청 실패: " + error.getMessage()));
    }

    public Mono<Map<String, Object>> fetchAnalysesByUser(String token) {
        return webClient.get()
                .uri("/fastapi/api/analysis-by-user/")
                .header("Authorization", "Bearer " + token)
                .retrieve()
                .onStatus(HttpStatusCode::is4xxClientError, clientResponse ->
                        Mono.error(new RuntimeException("FastAPI 요청 오류 (4xx): " + clientResponse.statusCode()))
                )
                .onStatus(HttpStatusCode::is5xxServerError, clientResponse ->
                        Mono.error(new RuntimeException("FastAPI 서버 오류 (5xx): " + clientResponse.statusCode()))
                )
                .bodyToMono(new ParameterizedTypeReference<Map<String, Object>>() {})
                .doOnError(error -> logger.error("🚨 FastAPI 요청 실패: " + error.getMessage()));
    }

    public Mono<Map<String, Object>> fetchAnalysisStats(String token) {
        return webClient.get()
                .uri("/fastapi/api/analysis/stats")
                .header("Authorization", "Bearer " + token)
                .retrieve()
                .onStatus(HttpStatusCode::is4xxClientError, clientResponse ->
                    Mono.error(new RuntimeException("FastAPI 요청 오류 (4xx): " + clientResponse.statusCode()))
                )
                .onStatus(HttpStatusCode::is5xxServerError, clientResponse ->
                    Mono.error(new RuntimeException("FastAPI 서버 오류 (5xx): " + clientResponse.statusCode()))
                )
                .bodyToMono(new ParameterizedTypeReference<Map<String, Object>>() {})
                .doOnError(error -> logger.error("🚨 FastAPI 요청 실패: " + error.getMessage()));
    }

    public Mono<Map<String, Object>> getAnalysis(String filename, String authorization) {
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
                .bodyToMono(new ParameterizedTypeReference<Map<String, Object>>() {})
                .doOnError(error -> logger.error("🚨 FastAPI 요청 실패: " + error.getMessage()));
    }

    public Mono<Map<String, Object>> analyzeScript(
        String filePath, String filename, String speechMinutes, String token
    ) {
        logger.info("📄 FastAPI로 스크립트 분석 요청: " + filePath);
        return webClient.post()
                .uri("/fastapi/api/analyze-script/")
                .header("Authorization", "Bearer " + token)
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(BodyInserters.fromMultipartData("file", new FileSystemResource(filePath))
                            .with("filename", filename)
                            .with("speech_minutes", speechMinutes)) // speech_minutes 추가
                .retrieve()
                .onStatus(HttpStatusCode::is4xxClientError, clientResponse ->
                        Mono.error(new RuntimeException("FastAPI 요청 오류 (4xx): " + clientResponse.statusCode()))
                )
                .onStatus(HttpStatusCode::is5xxServerError, clientResponse ->
                        Mono.error(new RuntimeException("FastAPI 서버 오류 (5xx): " + clientResponse.statusCode()))
                )
                .bodyToMono(new ParameterizedTypeReference<Map<String, Object>>() {})
                .doOnError(error -> logger.error("🚨 FastAPI 요청 실패: " + error.getMessage()));
    }

    public Mono<Map<String, Object>> analyzeTextScript(Map<String, Object> request, String token) {
        logger.info("📝 FastAPI로 텍스트 대본 분석 요청: " + request.get("script_text"));
        return webClient.post()
                .uri("/fastapi/api/analyze-text-script/")
                .header("Authorization", "Bearer " + token)
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(request)
                .retrieve()
                .onStatus(HttpStatusCode::is4xxClientError, clientResponse ->
                        Mono.error(new RuntimeException("FastAPI 요청 오류 (4xx): " + clientResponse.statusCode()))
                )
                .onStatus(HttpStatusCode::is5xxServerError, clientResponse ->
                        Mono.error(new RuntimeException("FastAPI 서버 오류 (5xx): " + clientResponse.statusCode()))
                )
                .bodyToMono(new ParameterizedTypeReference<Map<String, Object>>() {})
                .doOnError(error -> logger.error("🚨 FastAPI 요청 실패: " + error.getMessage()));
    }

    public Mono<Map<String, Object>> getScriptAnalysis(String scriptId, String token) {
        logger.info("🔍 FastAPI로 대본 분석 조회 요청: script_id=" + scriptId);
        return webClient.get()
                .uri(uriBuilder -> uriBuilder
                        .path("/fastapi/api/get-script-analysis/")
                        .queryParam("script_id", scriptId)
                        .build())
                .header("Authorization", "Bearer " + token)
                .retrieve()
                .onStatus(HttpStatusCode::is4xxClientError, clientResponse ->
                        Mono.error(new RuntimeException("FastAPI 요청 오류 (4xx): " + clientResponse.statusCode()))
                )
                .onStatus(HttpStatusCode::is5xxServerError, clientResponse ->
                        Mono.error(new RuntimeException("FastAPI 서버 오류 (5xx): " + clientResponse.statusCode()))
                )
                .bodyToMono(new ParameterizedTypeReference<Map<String, Object>>() {})
                .doOnError(error -> logger.error("🚨 FastAPI 요청 실패: " + error.getMessage()));
    }
}