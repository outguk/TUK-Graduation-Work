package TUK_Graduation_Work.GW_backend.controller;

import TUK_Graduation_Work.GW_backend.domain.User;
import TUK_Graduation_Work.GW_backend.service.UserService;
import TUK_Graduation_Work.GW_backend.util.JwtUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.context.ReactiveSecurityContextHolder;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Mono;

import java.util.HashMap;
import java.util.Map;

@CrossOrigin(origins = "http://localhost:5173")
@RestController
@RequestMapping("/spring/api")
public class UserController {

    private final UserService userService;
    private final JwtUtil jwtUtil;
    private final Logger logger = LoggerFactory.getLogger(UserController.class);

    @Autowired
    public UserController(UserService userService, JwtUtil jwtUtil) {
        this.userService = userService;
        this.jwtUtil = jwtUtil;
    }

    static class UserForm {
        private String username;
        private String password;
        private String email;

        public String getUsername() { return username; }
        public void setUsername(String username) { this.username = username; }
        public String getPassword() { return password; }
        public void setPassword(String password) { this.password = password; }
        public String getEmail() { return email; }
        public void setEmail(String email) { this.email = email; }
    }

    @PostMapping("/users/new")
    public Mono<ResponseEntity<Map<String, Object>>> create(@RequestBody Mono<UserForm> formMono) {
        return formMono.flatMap(form -> {
            if (form.getUsername() == null || form.getUsername().isEmpty() ||
                form.getEmail() == null || form.getEmail().isEmpty()) {
                Map<String, Object> error = new HashMap<>();
                error.put("message", "Username or Email cannot be null or empty");
                return Mono.just(ResponseEntity.badRequest().body(error));
            }

            User user = new User();
            user.setUsername(form.getUsername());
            user.setPassword(form.getPassword());
            user.setEmail(form.getEmail());

            logger.info("회원가입 요청 - username: {}, email: {}", user.getUsername(), user.getEmail());

            return userService.joinAsync(user)
                    .map(savedUser -> {
                        String token = jwtUtil.generateToken(savedUser.getId());
                        Map<String, Object> response = new HashMap<>();
                        response.put("message", "회원가입 성공 (RDS)");
                        response.put("userId", savedUser.getId());
                        response.put("username", savedUser.getUsername());
                        response.put("email", savedUser.getEmail());
                        response.put("createdAt", savedUser.getCreatedAt().toString());
                        response.put("token", token);
                        return ResponseEntity.ok(response);
                    })
                    .onErrorResume(e -> {
                        Map<String, Object> error = new HashMap<>();
                        error.put("message", "회원가입 실패: " + e.getMessage());
                        return Mono.just(ResponseEntity.badRequest().body(error));
                    });
        });
    }

    @PostMapping("/users/login")
    public Mono<ResponseEntity<Map<String, Object>>> checkLogin(@RequestBody Mono<UserForm> formMono) {
        return formMono.flatMap(form -> {
            Map<String, Object> response = new HashMap<>();
            if (form.getUsername() == null || form.getPassword() == null) {
                response.put("message", "아이디/비밀번호가 누락되었습니다");
                return Mono.just(ResponseEntity.badRequest().body(response));
            }

            return userService.findOneAsync(form.getUsername())
                    .flatMap(user -> userService.checkPassword(user.getId(), form.getPassword())
                        .flatMap(isMatch -> {
                            if (isMatch) {
                                String token = jwtUtil.generateToken(user.getId());
                                Map<String, Object> successMap = new HashMap<>();
                                successMap.put("message", "로그인 성공 (RDS)");
                                successMap.put("userId", user.getId());
                                successMap.put("email", user.getEmail());
                                successMap.put("createdAt", user.getCreatedAt().toString());
                                successMap.put("token", token);
                                return Mono.just(ResponseEntity.ok(successMap));
                            } else {
                                response.put("message", "로그인 실패 (비밀번호 불일치)");
                                return Mono.just(ResponseEntity.status(400).body(response));
                            }
                        }))
                    .switchIfEmpty(Mono.fromCallable(() -> {
                        response.put("message", "로그인 실패 (아이디 없음)");
                        return ResponseEntity.status(400).body(response);
                    }));
        });
    }

    @GetMapping("/user/profile")
    public Mono<ResponseEntity<Map<String, Object>>> getUserProfile() {
        return ReactiveSecurityContextHolder.getContext()
            .map(context -> (Long) context.getAuthentication().getPrincipal())
            .flatMap(userId -> userService.findByIdAsync(userId))
            .flatMap(user -> {
                Map<String, Object> response = new HashMap<>();
                response.put("name", user.getUsername());
                response.put("email", user.getEmail());
                response.put("joinDate", user.getCreatedAt().toString());
                logger.info("Profile fetched - userId: {}, name: {}, email: {}", user.getId(), user.getUsername(), user.getEmail());
                return Mono.just(ResponseEntity.ok(response));
            })
            .switchIfEmpty(Mono.just(ResponseEntity.status(401).body(Map.of("message", "User not found or invalid token"))));
    }

    static class ProfileUpdateForm {
        private String name;
        private String email;
        private String currentPassword;
        private String newPassword;

        public String getName() { return name; }
        public void setName(String name) { this.name = name; }
        public String getEmail() { return email; }
        public void setEmail(String email) { this.email = email; }
        public String getCurrentPassword() { return currentPassword; }
        public void setCurrentPassword(String currentPassword) { this.currentPassword = currentPassword; }
        public String getNewPassword() { return newPassword; }
        public void setNewPassword(String newPassword) { this.newPassword = newPassword; }
    }

    @PutMapping("/user/profile")
    public Mono<ResponseEntity<Map<String, Object>>> updateUserProfile(
            @RequestBody Mono<ProfileUpdateForm> formMono) {
        return ReactiveSecurityContextHolder.getContext()
            .map(context -> (Long) context.getAuthentication().getPrincipal())
            .flatMap(userId -> formMono.flatMap(form -> {
                Map<String, Object> response = new HashMap<>();

                if (form.getName() == null || form.getName().isEmpty()) {
                    response.put("message", "Name cannot be null or empty");
                    return Mono.just(ResponseEntity.badRequest().body(response));
                }

                logger.info("Update request - userId: {}, name: {}, email: {}, hasPassword: {}", 
                    userId, form.getName(), form.getEmail(), form.getCurrentPassword() != null);

                return userService.findByIdAsync(userId)
                    .switchIfEmpty(Mono.error(new IllegalArgumentException("User not found")))
                    .flatMap(user -> {
                        if (form.getCurrentPassword() != null && form.getNewPassword() != null) {
                            return userService.checkPassword(userId, form.getCurrentPassword())
                                .flatMap(isMatch -> {
                                    if (!isMatch) {
                                        response.put("message", "Current password incorrect");
                                        return Mono.just(ResponseEntity.status(401).body(response));
                                    }
                                    return userService.updateUserAsync(userId, form.getName(), form.getEmail(), form.getNewPassword())
                                        .map(updatedUser -> {
                                            response.put("message", "Profile updated successfully");
                                            response.put("name", updatedUser.getUsername());
                                            response.put("email", updatedUser.getEmail());
                                            logger.info("Profile updated - userId: {}, name: {}, email: {}", 
                                                userId, updatedUser.getUsername(), updatedUser.getEmail());
                                            return ResponseEntity.ok(response);
                                        });
                                });
                        }
                        return userService.updateUserAsync(userId, form.getName(), form.getEmail(), null)
                            .map(updatedUser -> {
                                response.put("message", "Profile updated successfully");
                                response.put("name", updatedUser.getUsername());
                                response.put("email", updatedUser.getEmail());
                                logger.info("Profile updated - userId: {}, name: {}, email: {}", 
                                    userId, updatedUser.getUsername(), updatedUser.getEmail());
                                return ResponseEntity.ok(response);
                            });
                    })
                    .onErrorResume(e -> {
                        response.put("message", "Profile update failed: " + e.getMessage());
                        if (e.getMessage().contains("Email already in use")) {
                            return Mono.just(ResponseEntity.status(409).body(response));
                        }
                        return Mono.just(ResponseEntity.status(500).body(response));
                    });
            }));
    }
}