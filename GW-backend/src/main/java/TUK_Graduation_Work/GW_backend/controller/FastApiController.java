package TUK_Graduation_Work.GW_backend.controller;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.http.MediaType;
import org.springframework.http.codec.multipart.FilePart;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ServerWebExchange;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;
import TUK_Graduation_Work.GW_backend.service.FastApiClient;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

@RestController
@CrossOrigin(origins = "http://localhost:5173")
public class FastApiController {

    private final FastApiClient fastApiClient;

    public FastApiController(FastApiClient fastApiClient) {
        this.fastApiClient = fastApiClient;
    }

    @PostMapping(value = "/upload", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public Mono<Map<String, Object>> handleFileUpload(
            @RequestPart("file") FilePart file,
            ServerWebExchange exchange
    ) {
        // WebFlux 세션 접근
        return exchange.getSession().flatMap(webSession -> {
            Object userIdObj = webSession.getAttribute("userId");
            if (userIdObj == null) {
                Map<String, Object> errorMap = new HashMap<>();
                errorMap.put("message", "로그인 필요: 세션에 userId가 없습니다.");
                return Mono.just(errorMap);
            }
            Long userId = (Long) userIdObj;

            // 임시 디렉토리에 파일 저장할 경로
            String tempDir = System.getProperty("java.io.tmpdir");
            String fileName = file.filename();
            Path tempPath = Paths.get(tempDir, fileName);

            // file.transferTo(...)는 이미 Mono<Void> 반환
            // 별도의 block() 없이 체인식으로 사용
            return file.transferTo(tempPath)
                // 파일 시스템 접근은 잠재적으로 블로킹이므로 별도 스레드풀에서 실행
                .publishOn(Schedulers.boundedElastic())
                .thenReturn(tempPath)  // Mono<Path>
                .flatMap(savedPath -> fastApiClient.uploadFileToFastAPI(savedPath.toString(), userId))
                .flatMap(jsonString -> {
                    try {
                        ObjectMapper objectMapper = new ObjectMapper();
                        Map<String, Object> resultMap = objectMapper.readValue(
                                jsonString, new TypeReference<Map<String, Object>>() {});
                        resultMap.put("message", "파일 업로드 및 분석 성공! (userId=" + userId + ")");
                        return Mono.just(resultMap);
                    } catch (Exception e) {
                        Map<String, Object> errorMap = new HashMap<>();
                        errorMap.put("message", "JSON 파싱 에러: " + e.getMessage());
                        return Mono.just(errorMap);
                    }
                })
                .onErrorResume(e -> {
                    Map<String, Object> errorMap = new HashMap<>();
                    errorMap.put("message", "파일 업로드 실패: " + e.getMessage());
                    return Mono.just(errorMap);
                });
        });
    }

     // --- 추가: 사용자별 분석 결과 조회 ---
    // GET /my-analyses -> 세션에서 userId -> fastApiClient.fetchAnalysesByUser(userId)
    @GetMapping("/my-analyses")
    public Mono<Map<String, Object>> getMyAnalyses(ServerWebExchange exchange) {
        return exchange.getSession().flatMap(webSession -> {
            Object userIdObj = webSession.getAttribute("userId");
            if (userIdObj == null) {
                Map<String, Object> errorMap = new HashMap<>();
                errorMap.put("message", "로그인 필요 (세션에 userId 없음)");
                return Mono.just(errorMap);
            }
            Long userId = (Long) userIdObj;
            return fastApiClient.fetchAnalysesByUser(userId);
        });
    }
}
