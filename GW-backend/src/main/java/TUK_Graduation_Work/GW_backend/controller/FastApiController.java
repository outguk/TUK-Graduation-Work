package TUK_Graduation_Work.GW_backend.controller;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.http.codec.multipart.FilePart;
import org.springframework.core.io.Resource;
import org.springframework.core.io.PathResource;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ServerWebExchange;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;
import TUK_Graduation_Work.GW_backend.service.FastApiClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpHeaders;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

@RestController
@CrossOrigin(origins = "http://localhost:5173")
@RequestMapping("/spring/api")
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
        String token = authorizationHeader.startsWith("Bearer ") ? authorizationHeader.substring(7) : authorizationHeader;

        String tempDir = System.getProperty("java.io.tmpdir");
        String fileName = file.filename();
        Path tempPath = Paths.get(tempDir, fileName);

        return file.transferTo(tempPath)
                .publishOn(Schedulers.boundedElastic())
                .thenReturn(tempPath)
                .flatMap(savedPath -> fastApiClient.uploadFileToFastAPI(savedPath.toString(), token))
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
    public Mono<Map<String, Object>> getAnalysis(
            @RequestParam("filename") String filename,
            @RequestHeader("Authorization") String authorization
    ) {
        logger.info("Fetching analysis for filename: {}", filename);
        return fastApiClient.getAnalysis(filename, authorization)
                .map(response -> {
                    logger.info("Successfully fetched analysis for filename: {}", filename);
                    Map<String, Object> result = new HashMap<>(response);
                    result.put("message", "분석 데이터 조회 성공");
                    return result;
                })
                .onErrorResume(e -> {
                    logger.error("Failed to fetch analysis for filename {}: {}", filename, e.getMessage());
                    return Mono.just(Map.of(
                            "error", "분석 데이터 조회 실패: " + e.getMessage(),
                            "message", "분석 데이터 조회 실패"
                    ));
                });
    }

    @GetMapping("/video/{filename:.+}")
    public Mono<ResponseEntity<Resource>> streamVideo(
            @PathVariable String filename,
            @RequestHeader HttpHeaders headers
    ) {
        Path videoPath = Paths.get(System.getProperty("user.home"), "videos", filename);
        Resource videoResource = new PathResource(videoPath);

        if (!videoResource.exists()) {
            return Mono.just(ResponseEntity.notFound().build());
        }

        return Mono.just(
                ResponseEntity.ok()
                        .header(HttpHeaders.CONTENT_TYPE, "video/mp4")
                        .header(HttpHeaders.ACCEPT_RANGES, "bytes")
                        .body(videoResource)
        );
    }

    @PostMapping(value = "/script-upload", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public Mono<Map<String, Object>> handleScriptUpload(
        @RequestPart("scriptFile") FilePart file,
        @RequestPart("filename") String filename,
        @RequestPart("speech_minutes") String speechMinutes,
        @RequestHeader("Authorization") String authHeader
    ) {
        String token = authHeader.replace("Bearer ", "");
        String tempDir = System.getProperty("java.io.tmpdir");
        Path tempPath = Paths.get(tempDir, file.filename());

        logger.info("대본 업로드 요청 - filename: {}, speech_minutes: {}", filename, speechMinutes);

        return file.transferTo(tempPath)
                .publishOn(Schedulers.boundedElastic())
                .then(fastApiClient.analyzeScript(tempPath.toString(), filename, speechMinutes, token))
                .map(response -> {
                    logger.info("FastAPI 응답 수신: {}", response);
                    Map<String, Object> result = new HashMap<>(response);
                    result.put("message", "대본 분석 성공");
                    
                    // _id와 id 필드 모두 확인하여 제공
                    if (response.containsKey("_id") && !response.containsKey("id")) {
                        result.put("id", response.get("_id"));
                    }
                    
                    logger.info("최종 응답: {}", result);
                    return result;
                })
                .onErrorResume(e -> {
                    logger.error("Failed to analyze script: {}", e.getMessage());
                    return Mono.just(Map.of(
                            "error", "대본 분석 실패: " + e.getMessage(),
                            "message", "대본 분석 실패"
                    ));
                });
    }

    @PostMapping(value = "/analyze-text-script", consumes = MediaType.APPLICATION_JSON_VALUE)
    public Mono<Map<String, Object>> analyzeTextScript(
            @RequestBody Map<String, Object> request,
            @RequestHeader("Authorization") String authHeader
    ) {
        logger.info("Processing text script analysis request: {}", request.get("script_text"));
        String token = authHeader.replace("Bearer ", "");
        return fastApiClient.analyzeTextScript(request, token)
                .map(response -> {
                    logger.info("Successfully analyzed text script");
                    Map<String, Object> result = new HashMap<>(response);
                    result.put("message", "대본 분석 성공");
                    return result;
                })
                .onErrorResume(e -> {
                    logger.error("Failed to analyze text script: {}", e.getMessage());
                    return Mono.just(Map.of(
                            "error", "대본 분석 실패: " + e.getMessage(),
                            "message", "대본 분석 실패"
                    ));
                });
    }

    @GetMapping("/get-script-analysis")
    public Mono<Map<String, Object>> getScriptAnalysis(
        @RequestParam("script_id") String scriptId,
        @RequestHeader("Authorization") String authHeader
    ) {
        logger.info("Fetching script analysis for script_id: {}", scriptId);
        String token = authHeader.replace("Bearer ", "");
        return fastApiClient.getScriptAnalysis(scriptId, token)
                .map(response -> {
                    logger.info("Successfully fetched script analysis for script_id: {}", scriptId);
                    Map<String, Object> result = new HashMap<>(response);
                    result.put("message", "대본 분석 결과 조회 성공");
                    return result;
                })
                .onErrorResume(e -> {
                    logger.error("Failed to fetch script analysis for script_id {}: {}", scriptId, e.getMessage());
                    return Mono.just(Map.of(
                            "error", "대본 분석 결과 조회 실패: " + e.getMessage(),
                            "message", "대본 분석 결과 조회 실패"
                    ));
                });
    }
}