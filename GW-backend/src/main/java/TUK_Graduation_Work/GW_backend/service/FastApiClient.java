package TUK_Graduation_Work.GW_backend.service;

import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Service;
import org.springframework.http.MediaType;
import org.springframework.core.io.FileSystemResource;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;

import java.io.File;
import java.util.Map;

@Service
public class FastApiClient {
    private final WebClient webClient = WebClient.builder().baseUrl("http://localhost:5000").build();

    /**
     * 파일 업로드 요청 시, userId를 쿼리 파라미터로 추가하여 FastAPI의 /upload-video 엔드포인트에 전송합니다.
     *
     * @param filePath 로컬에 저장된 파일 경로
     * @param userId   로그인한 사용자의 ID
     * @return FastAPI 응답 (JSON 문자열)
     */
    public Mono<String> uploadFileToFastAPI(String filePath, Long userId) {
        File videoFile = new File(filePath);
        if (!videoFile.exists() || !videoFile.isFile()) {
            return Mono.error(new RuntimeException("Error: 업로드할 파일을 찾을 수 없습니다!"));
        }

        System.out.println("📂 FastAPI로 업로드 요청: " + filePath + ", userId: " + userId);

        FileSystemResource fileResource = new FileSystemResource(videoFile);

        return webClient.post()
                .uri(uriBuilder -> uriBuilder
                        .path("/upload-video/")
                        .queryParam("user_id", userId)
                        .build())
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(BodyInserters.fromMultipartData("file", fileResource))
                .retrieve()
                .onStatus(HttpStatusCode::is4xxClientError, clientResponse ->
                        Mono.error(new RuntimeException("FastAPI 요청 오류 (4xx): " + clientResponse.statusCode())))
                .onStatus(HttpStatusCode::is5xxServerError, clientResponse ->
                        Mono.error(new RuntimeException("FastAPI 서버 오류 (5xx): " + clientResponse.statusCode())))
                .bodyToMono(String.class)
                .doOnError(error -> System.err.println("🚨 FastAPI 요청 실패: " + error.getMessage()));
    }

    // --- 추가: 사용자별 분석 결과 조회 ---
    // GET /analysis-by-user?user_id=xxx -> {"analyses": [...]} 형식
    public Mono<Map<String,Object>> fetchAnalysesByUser(Long userId) {
        return webClient.get()
                .uri(uriBuilder -> uriBuilder
                    .path("/analysis-by-user/")
                    .queryParam("user_id", userId)
                    .build())
                .retrieve()
                .onStatus(HttpStatusCode::is4xxClientError, clientResponse ->
                    Mono.error(new RuntimeException("FastAPI 요청 오류 (4xx): " + clientResponse.statusCode()))
                )
                .onStatus(HttpStatusCode::is5xxServerError, clientResponse ->
                    Mono.error(new RuntimeException("FastAPI 서버 오류 (5xx): " + clientResponse.statusCode()))
                )
                // 응답이 {"analyses": [ ... ]} 형식이라 가정
                .bodyToMono(new org.springframework.core.ParameterizedTypeReference<Map<String,Object>>() {})
                .doOnError(error -> System.err.println("🚨 FastAPI 요청 실패: " + error.getMessage()));
    }
}