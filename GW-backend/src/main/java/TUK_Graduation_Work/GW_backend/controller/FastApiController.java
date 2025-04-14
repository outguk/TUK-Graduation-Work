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
@RequestMapping("/spring/api") // 경로 통일
public class FastApiController {

    private final FastApiClient fastApiClient;

    public FastApiController(FastApiClient fastApiClient) {
        this.fastApiClient = fastApiClient;
    }

    @PostMapping(value = "/upload", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public Mono<Map<String, Object>> handleFileUpload(
            @RequestPart("file") FilePart file,
            @RequestHeader("Authorization") String authorizationHeader,
            ServerWebExchange exchange
    ) {
        // JWT 토큰 추출
        String token = authorizationHeader.startsWith("Bearer ") ? authorizationHeader.substring(7) : authorizationHeader;

        // 임시 디렉토리에 파일 저장
        String tempDir = System.getProperty("java.io.tmpdir");
        String fileName = file.filename();
        Path tempPath = Paths.get(tempDir, fileName);

        return file.transferTo(tempPath)
                .publishOn(Schedulers.boundedElastic()) // 파일 I/O를 별도 스레드에서 처리
                .thenReturn(tempPath)  // Mono<Path>
                .flatMap(savedPath -> fastApiClient.uploadFileToFastAPI(savedPath.toString(), token)) // token 사용
                .flatMap(jsonString -> {
                    try {
                        ObjectMapper objectMapper = new ObjectMapper();
                        Map<String, Object> resultMap = objectMapper.readValue(
                                jsonString, new TypeReference<Map<String, Object>>() {});
                        resultMap.put("message", "파일 업로드 및 분석 성공!");
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
    }

    @GetMapping("/my-analyses")
    public Mono<Map<String, Object>> getMyAnalyses(
            @RequestHeader("Authorization") String authorizationHeader,
            ServerWebExchange exchange
    ) {
        // JWT 토큰 추출
        String token = authorizationHeader.startsWith("Bearer ") ? authorizationHeader.substring(7) : authorizationHeader;

        return fastApiClient.fetchAnalysesByUser(token)
                .map(result -> {
                    result.put("message", "분석 결과 조회 성공");
                    return result;
                })
                .onErrorResume(e -> {
                    Map<String, Object> errorMap = new HashMap<>();
                    errorMap.put("message", "분석 결과 조회 실패: " + e.getMessage());
                    return Mono.just(errorMap);
                });
    }
}