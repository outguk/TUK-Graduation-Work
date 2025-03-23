package TUK_Graduation_Work.GW_backend.controller;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.http.MediaType;
import org.springframework.http.codec.multipart.FilePart;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Mono;
import TUK_Graduation_Work.GW_backend.service.FastApiClient;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

@Controller
public class FastApiController {

    private final FastApiClient fastApiClient;

    public FastApiController(FastApiClient fastApiClient) {
        this.fastApiClient = fastApiClient;
    }

    @GetMapping("/upload")
    public String fileUploadPage() {
        return "upload";
    }

    @PostMapping(value = "/upload", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public Mono<String> handleFileUpload(@RequestPart("file") FilePart file, Model model) {
        String tempDir = System.getProperty("java.io.tmpdir");
        String fileName = file.filename();
        Path tempPath = Paths.get(tempDir, fileName);

        // 파일을 임시 디렉토리에 저장
        return file.transferTo(tempPath)
                // FastAPI로 업로드 및 분석 요청
                .then(fastApiClient.uploadFileToFastAPI(tempPath.toString()))
                // JSON 문자열을 받아서 Jackson으로 파싱
                .flatMap(jsonString -> {
                    try {
                        ObjectMapper objectMapper = new ObjectMapper();
                        // JSON을 Map<String, Object> 형태로 파싱
                        Map<String, Object> resultMap = objectMapper.readValue(
                                jsonString, new TypeReference<Map<String, Object>>() {}
                        );

                        // resultMap 예시:
                        // {
                        //   "filename": "video.mp4",
                        //   "speaking_speed": { ... },
                        //   "volume_analysis": { ... }
                        // }

                        // 원하는 키값을 꺼내서 Model에 담아둔다
                        model.addAttribute("message", "파일 업로드 및 분석 성공!");
                        model.addAttribute("analysisFilename", resultMap.get("filename"));
                        model.addAttribute("speakingSpeed", resultMap.get("speaking_speed"));
                        model.addAttribute("speakingEvaluation", resultMap.get("speaking_evaluation"));
                        model.addAttribute("volumeAnalysis", resultMap.get("volume_analysis"));
                        model.addAttribute("volumeEvaluation", resultMap.get("volume_evaluation"));
                        model.addAttribute("nonverbalAnalysis", resultMap.get("nonverbal_analysis"));

                        // "upload"라는 Thymeleaf 템플릿으로 이동
                        return Mono.just("upload");

                    } catch (Exception e) {
                        // 파싱 실패 시
                        model.addAttribute("message", "JSON 파싱 에러: " + e.getMessage());
                        return Mono.just("upload");
                    }
                })
                .onErrorResume(e -> {
                    model.addAttribute("message", "파일 업로드 실패: " + e.getMessage());
                    return Mono.just("upload");
                });
    }
}