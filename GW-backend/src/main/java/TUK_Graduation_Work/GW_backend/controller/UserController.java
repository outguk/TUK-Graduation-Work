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

    @PostMapping("/users/new")
    public String create(@RequestBody UserForm form) {
        User user = new User();
        user.setName(form.getName());
        user.setPassword(form.getPassword());

        logger.info("member: {}", user.getName());
        logger.info("password: {}", user.getPassword());

        // 회원 가입 전 중복 체크, 암호화 등의 로직 추가 고려
        userService.join(user);

        return "회원가입 성공";
    }

    @PostMapping("/users/login")
    public String checkLogin(@RequestBody UserForm form) {
        Optional<User> userOpt = userService.findOne(form.getName());
        if (userOpt.isPresent()) {
            User user = userOpt.get();
            if (form.getPassword().equals(user.getPassword())) {
                return "로그인 성공";
            }
        }
        return "로그인 실패";
    }
}
