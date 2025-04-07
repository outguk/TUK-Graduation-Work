package TUK_Graduation_Work.GW_backend.controller;

import TUK_Graduation_Work.GW_backend.domain.User;
import TUK_Graduation_Work.GW_backend.service.UserService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.server.ServerWebExchange;
import reactor.core.publisher.Mono;

import java.util.HashMap;
import java.util.Map;

@CrossOrigin(origins = "http://localhost:5173")
@RestController
public class UserController {

    private final UserService userService;
    private final Logger logger = LoggerFactory.getLogger(UserController.class);

    public UserController(UserService userService) {
        this.userService = userService;
    }

    // DTO: 클라이언트에서 username/password를 받는다
    static class UserForm {
        private String username;
        private String password;

        public String getUsername() { return username; }
        public void setUsername(String username) { this.username = username; }
        public String getPassword() { return password; }
        public void setPassword(String password) { this.password = password; }
    }

    // 회원가입 (/users/new)
    //  - formMono: JSON 바디 (username, password)
    @PostMapping("/users/new")
    public Mono<ResponseEntity<Map<String, Object>>> create(@RequestBody Mono<UserForm> formMono) {
        return formMono.flatMap(form -> {
            if (form.getUsername() == null || form.getUsername().isEmpty()) {
                Map<String, Object> error = new HashMap<>();
                error.put("message", "Username cannot be null or empty");
                return Mono.just(ResponseEntity.badRequest().body(error));
            }

            // User 엔티티 생성
            User user = new User();
            user.setUsername(form.getUsername());
            user.setPassword(form.getPassword());

            logger.info("회원가입 요청 - username: {}", user.getUsername());

            // userService.join(...)는 블로킹이므로, 비동기로 감싸야 함
            return userService.joinAsync(user)
                    .map(savedUser -> {
                        Map<String, Object> response = new HashMap<>();
                        response.put("message", "회원가입 성공 (RDS)");
                        response.put("userId", savedUser.getId());
                        return ResponseEntity.ok(response);
                    })
                    .onErrorResume(e -> {
                        // 중복 에러 등 처리
                        Map<String, Object> error = new HashMap<>();
                        error.put("message", "회원가입 실패: " + e.getMessage());
                        return Mono.just(ResponseEntity.badRequest().body(error));
                    });
        });
    }

    // 로그인 (/users/login)
    //  - 비동기 formMono + ServerWebExchange로 세션 접근
    @PostMapping("/users/login")
    public Mono<ResponseEntity<Map<String, Object>>> checkLogin(
            @RequestBody Mono<UserForm> formMono,
            ServerWebExchange exchange
    ) {
        return formMono.flatMap(form -> {
            Map<String, Object> response = new HashMap<>();
            if (form.getUsername() == null || form.getPassword() == null) {
                response.put("message", "아이디/비밀번호가 누락되었습니다");
                return Mono.just(ResponseEntity.badRequest().body(response));
            }

            // DB에서 사용자 조회 (비동기)
            return userService.findOneAsync(form.getUsername())
                    .flatMap(user -> {
                        // 찾았다면 비밀번호 확인
                        if (form.getPassword().equals(user.getPassword())) {
                            // 세션에 userId 저장
                            return exchange.getSession().flatMap(webSession -> {
                                webSession.getAttributes().put("userId", user.getId());

                                Map<String, Object> successMap = new HashMap<>();
                                successMap.put("message", "로그인 성공 (RDS)");
                                successMap.put("userId", user.getId());
                                return Mono.just(ResponseEntity.ok(successMap));
                            });
                        } else {
                            response.put("message", "로그인 실패 (비밀번호 불일치)");
                            return Mono.just(ResponseEntity.status(400).body(response));
                        }
                    })
                    .switchIfEmpty(Mono.fromCallable(() -> {
                        // 사용자 없음
                        response.put("message", "로그인 실패 (아이디 없음)");
                        return ResponseEntity.status(400).body(response);
                    }));
        });
    }
}
