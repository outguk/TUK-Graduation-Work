package TUK_Graduation_Work.GW_backend.controller;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.http.MediaType;
import org.springframework.http.codec.multipart.FilePart;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Mono;
import TUK_Graduation_Work.GW_backend.service.FastApiClient;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

 /**
     * 변경점
     * 1. @RestController 사용 RESTful API 응답을 위한 컨트롤러 선언
     * 2. React는 Json 형식의 데이터를 사용하므로 분석 결과를 Json 형식으로 받아 프론트엔드에 넘겨주도록 변경
     * 3. 기존 타임리프 템플릿을 통해 UI를 구성하지 않으므로 반환값으로 주소를 반환하지 않고, @GetMapping 부분을 삭제함
     * 4. @CrossOrigin(origins = "http://localhost:5173")를 통해 프론트엔드에서 요청을 허용하도록 설정
     */

@RestController
@CrossOrigin(origins = "http://localhost:5173")
public class FastApiController {

    private final FastApiClient fastApiClient;

    public FastApiController(FastApiClient fastApiClient) {
        this.fastApiClient = fastApiClient;
    }

    @PostMapping(value = "/upload", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public Mono<Map<String, Object>> handleFileUpload(@RequestPart("file") FilePart file) {
        String tempDir = System.getProperty("java.io.tmpdir");
        String fileName = file.filename();
        Path tempPath = Paths.get(tempDir, fileName);

        return file.transferTo(tempPath)
                .then(fastApiClient.uploadFileToFastAPI(tempPath.toString()))
                .flatMap(jsonString -> {
                    try {
                        ObjectMapper objectMapper = new ObjectMapper();
                        // JSON 문자열을 Map 형태로 파싱
                        Map<String, Object> resultMap = objectMapper.readValue(
                                jsonString, new TypeReference<Map<String, Object>>() {}
                        );
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
}
