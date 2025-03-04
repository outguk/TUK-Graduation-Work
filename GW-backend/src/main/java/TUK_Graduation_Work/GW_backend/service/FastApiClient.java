package TUK_Graduation_Work.GW_backend.service;

import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Service;
import org.springframework.http.MediaType;
import org.springframework.core.io.FileSystemResource;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Mono;


import java.io.File;

@Service
public class FastApiClient {
    private final WebClient webClient = WebClient.builder().baseUrl("http://localhost:5000").build();

    public Mono<String> uploadFileToFastAPI(String filePath) {
        File videoFile = new File(filePath);
        if (!videoFile.exists() || !videoFile.isFile()) {
            return Mono.error(new RuntimeException("Error: 업로드할 파일을 찾을 수 없습니다!"));
        }

        System.out.println("📂 FastAPI로 업로드 요청: " + filePath);

        FileSystemResource fileResource = new FileSystemResource(videoFile);

        return webClient.post()
                .uri("/upload-video/")
                .contentType(MediaType.MULTIPART_FORM_DATA)
                .body(BodyInserters.fromMultipartData("file", fileResource))  // ✅ multipart 설정
                .retrieve()

                // HTTP 4xx 및 5xx 상태 코드 예외 처리
                .onStatus(HttpStatusCode::is4xxClientError, clientResponse ->
                        Mono.error(new RuntimeException("FastAPI 요청 오류 (4xx): " + clientResponse.statusCode())))
                .onStatus(HttpStatusCode::is5xxServerError, clientResponse ->
                        Mono.error(new RuntimeException("FastAPI 서버 오류 (5xx): " + clientResponse.statusCode())))

                .bodyToMono(String.class)
                .doOnError(error -> System.err.println("🚨 FastAPI 요청 실패: " + error.getMessage()));  // ✅ 추가적인 예외 처리

    }

}