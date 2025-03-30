package TUK_Graduation_Work.GW_backend.controller;

import TUK_Graduation_Work.GW_backend.domain.User;
import TUK_Graduation_Work.GW_backend.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;

 /**
     * 변경점
     * 1. @RestController 사용 RESTful API 응답을 위한 컨트롤러 선언
     * 2. React에서는 Json 형식 데이터를 사용하므로 @ModelAttribute 대신 @RequestBody를 사용
     * 3. 기존 타임리프 템플릿을 통해 UI를 구성하지 않으므로 반환값으로 주소를 반환하지 않고, @GetMapping 부분을 삭제함
     * 4. @CrossOrigin(origins = "http://localhost:5173")를 통해 프론트엔드에서 요청을 허용하도록 설정
     */

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
