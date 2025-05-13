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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.core.io.Resource;
import org.springframework.core.io.PathResource;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

@RestController
@CrossOrigin(origins = "http://localhost:5173")
@RequestMapping("/spring/api") // 경로 통일
public class FastApiController {

    private static final Logger logger = LoggerFactory.getLogger(FastApiController.class);
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

    @GetMapping("/get-analysis")
    public Mono<Map> getAnalysis(@RequestParam("filename") String filename,
                                                @RequestHeader("Authorization") String authorization) {
        logger.info("Fetching analysis for filename: {}", filename);
        return fastApiClient.getAnalysis(filename, authorization)
                .map(response -> {
                    logger.info("Successfully fetched analysis for filename: {}", filename);
                    return response;
                })
                .onErrorResume(e -> {
                    logger.error("Failed to fetch analysis for filename {}: {}", filename, e.getMessage());
                    return Mono.just(Map.of(
                            "error", "분석 데이터 조회 실패: " + e.getMessage()
                    )).cast(Map.class);
                });
    }

    @GetMapping("/video/{filename:.+}")
    public Mono<ResponseEntity<Resource>> streamVideo(
            @PathVariable String filename,
            @RequestHeader HttpHeaders headers
    ) {
        // 서버에 비디오가 저장된 디렉토리 경로
        Path videoPath = Paths.get(System.getProperty("user.home"), "videos", filename);
        Resource videoResource = new PathResource(videoPath);

        if (!videoResource.exists()) {
            return Mono.just(ResponseEntity.notFound().build());
        }

        // 간단히 전체 파일을 반환 (Range 요청 처리 없이)
        return Mono.just(
                ResponseEntity.ok()
                        .header(HttpHeaders.CONTENT_TYPE, "video/mp4")
                        .header(HttpHeaders.ACCEPT_RANGES, "bytes")
                        .body(videoResource)
        );
    }
    // 스크립트 업로드
    @PostMapping(value = "/script-upload", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public Mono<Map<String,Object>> handleScriptUpload(
        @RequestPart("scriptFile") FilePart file,
        @RequestPart("filename") String filename,
        @RequestHeader("Authorization") String authHeader
    ) {
        String token = authHeader.replace("Bearer ", "");
        return file.transferTo(
            Paths.get(System.getProperty("java.io.tmpdir"), file.filename())
        )
        .then(fastApiClient.analyzeScript(
            Paths.get(System.getProperty("java.io.tmpdir"), file.filename()).toString(),
            filename,
            token
        ));
    }
  




}