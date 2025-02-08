package hello.hellospring.service;

import org.springframework.http.*;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;
import org.springframework.web.client.RestTemplate;
import org.springframework.core.io.FileSystemResource;
import java.io.File;
import java.util.Scanner;

public class FastApiClient {
    public static void main(String[] args) {
        // 사용자 입력을 통한 파일 경로 설정
        Scanner scanner = new Scanner(System.in);
        System.out.print("Enter the video file path: ");
        String filePath = scanner.nextLine();
        scanner.close();

        // 파일 존재 여부 확인
        File videoFile = new File(filePath);
        if (!videoFile.exists()) {
            System.err.println("Error: Video file not found!");
            return;
        }

        // FastAPI 서버 URL
        String fastApiUrl = "http://localhost:5000/upload-video/";

        // 파일 업로드 요청 생성
        FileSystemResource fileResource = new FileSystemResource(videoFile);
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);

        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("file", fileResource);

        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);
        RestTemplate restTemplate = new RestTemplate();

        // 요청 전송 및 예외 처리 추가
        try {
            ResponseEntity<String> response = restTemplate.exchange(fastApiUrl, HttpMethod.POST, requestEntity, String.class);

            if (response.getStatusCode() == HttpStatus.OK) {
                System.out.println("Response from FastAPI: " + response.getBody());
            } else {
                System.err.println("Error: Server returned " + response.getStatusCode());
            }
        } catch (Exception e) {
            System.err.println("Error while sending request: " + e.getMessage());
        }
    }
}

