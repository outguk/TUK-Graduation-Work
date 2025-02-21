package TUK_Graduation_Work.GW_backend.controller;

import TUK_Graduation_Work.GW_backend.domain.User;
import TUK_Graduation_Work.GW_backend.service.UserService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.RequestParam;

import java.util.List;
import java.util.Optional;

@Controller
public class UserController {
    private final UserService userService;

    @Autowired
    public UserController(UserService userService) {
        this.userService = userService;
    }

    @GetMapping("/users/new")
    public String createForm() {
        return "users/createUserForm";
    }

    @PostMapping("/users/new")
    public String create(UserForm form) {
        User user = new User();
        user.setName(form.getName());
        user.setPassword(form.getPassword());

        System.out.println("member " + user.getName());
        System.out.println("password " + user.getPassword());

        userService.join(user);

        return "redirect:/"; // home 화면으로 돌아감
    }

    @GetMapping("/users")
    public String list(Model model) {
        List<User> users = userService.findUsers();
        model.addAttribute("users", users);
        return "users/userList";
    }

    @PostMapping("/users/login")
    public String checkLogin(UserForm form, Model model) { // login 검증 (파라미터 두개 가능)
       if(userService.findOne(form.getName()).isPresent()){ // form.getName()이 존재하고
            Optional<User> user = userService.findOne(form.getName());


            // 그에 해당하는 비밀번호가 입력된 비밀번호와 일치하면
            if(form.getPassword().equals(user.map(User::getPassword).orElse(null))){
                model.addAttribute("name", form.getName());
                return "upload"; // welcome 페이지로 이동
            }
       }
       return "loginError";
    }
}

