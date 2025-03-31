package TUK_Graduation_Work.GW_backend.controller;

import TUK_Graduation_Work.GW_backend.domain.User;
import TUK_Graduation_Work.GW_backend.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;

@CrossOrigin(origins = "http://localhost:5173")
@RestController
public class UserController {

    private final UserService userService;
    private final Logger logger = LoggerFactory.getLogger(UserController.class);

    @Autowired
    public UserController(UserService userService) {
        this.userService = userService;
    }

    // DTO 요청
    static class UserForm {
        private String name;
        private String password;
        // getter/setter
        public String getName() { return name; }
        public void setName(String name) { this.name = name; }
        public String getPassword() { return password; }
        public void setPassword(String password) { this.password = password; }
    }

    // 회원가입
    @PostMapping("/users/new")
    public String create(@RequestBody UserForm form) {
        User user = new User();
        user.setName(form.getName());
        user.setPassword(form.getPassword());

        logger.info("member: {}", user.getName());
        logger.info("password: {}", user.getPassword());

        // DB 저장
        userService.join(user);
        return "회원가입 성공 (RDS)";
    }

    // 로그인
    @PostMapping("/users/login")
    public String checkLogin(@RequestBody UserForm form) {
        Optional<User> userOpt = userService.findOne(form.getName());
        if (userOpt.isPresent()) {
            User user = userOpt.get();
            if (form.getPassword().equals(user.getPassword())) {
                return "로그인 성공 (RDS)";
            }
        }
        return "로그인 실패";
    }
}
