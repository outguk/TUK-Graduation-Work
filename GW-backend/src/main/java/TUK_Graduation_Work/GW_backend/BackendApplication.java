package TUK_Graduation_Work.GW_backend;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class BackendApplication {

    public static void main(String[] args) {
        System.out.println("✅ Application starting...");  // 추가
        SpringApplication.run(BackendApplication.class, args);
    }
}
