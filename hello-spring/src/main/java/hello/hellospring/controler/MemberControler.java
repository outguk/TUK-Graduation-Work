package hello.hellospring.controler;

import hello.hellospring.domain.Member;
import hello.hellospring.service.MemberService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.ui.Model;

import java.util.List;

@Controller // spring에서 Controller 객체를 생성해 관리해줌 -> 빈을 관리한다.
public class MemberControler {

    private final MemberService memberService;

    @Autowired // 아래 memberService와 컨테이너에 있는 MemberService를 연결
    public MemberControler(MemberService memberService) {
        this.memberService = memberService;
    }

    @GetMapping("/members/new")
    public String createForm() {
        return "members/createMemberForm";
    }

    @PostMapping("/members/new")
    public String create(MemberForm form) {
        Member member = new Member();
        member.setName(form.getName());

        memberService.join(member);
        System.out.println("member = " + member.getName());

        return "redirect:/"; // home 화면으로 돌아감
    }

    @GetMapping("/members")
    public String list(Model model) {
        List<Member> members = memberService.findMembers();
        model.addAttribute("members", members);
        return "members/memberList";
    }
}
