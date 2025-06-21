package TUK_Graduation_Work.GW_backend.controller;

import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@CrossOrigin(origins = "http://localhost:5173")  // React와 연결
@RestController
public class HomeController {

    @GetMapping("/api/home")
    public String home(){
        return "Welcome";
    }
}

