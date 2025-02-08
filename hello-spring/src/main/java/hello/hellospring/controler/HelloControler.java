package hello.hellospring.controler;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.ResponseBody;

@Controller
public class HelloControler {

    @GetMapping("hello") // /hello 가 들어오면 아래 메서드 호출
    public String hello(Model model){ // 스프링이 모델을 만들어 넣어줌
        model.addAttribute("data","hello!!"); // -> model(data : hello) 매핑
        return "hello"; // 템플릿에서 hello.html 를 찾아 실행해라
    }
    @GetMapping("hello-mvc")
    public String helloMvc(@RequestParam("name") String name, Model model){
        model.addAttribute("name",name);
        return "hello-template";
    }

    @GetMapping("hello-string")
    @ResponseBody
    public String helloString(@RequestParam("name") String name){
        return "hello" + name;
    }

    @GetMapping("hello-api")
    @ResponseBody
    public Hello helloApi(@RequestParam("name") String name){
        Hello hello = new Hello();
        hello.setName(name);
        return hello; // JSON 으로 반환이 기본 세팅
    }

    static class Hello{ // 객체 생성
        private String name;

        public String getName(){ // 넣을 때
            return name;
        }

        public void setName(String name){ // 꺼낼 때
            this.name = name;
        }
    }

}

