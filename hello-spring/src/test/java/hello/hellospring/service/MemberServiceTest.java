package hello.hellospring.service;

import hello.hellospring.domain.Member;
import hello.hellospring.repository.MemoryMemberRepository;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

class MemberServiceTest {

    MemberService memberService;
    MemoryMemberRepository memoryMemberRepository;

    @BeforeEach // 각 실행 이전에
    public void beforEach() {
        memberService = new MemberService(memoryMemberRepository);
        memoryMemberRepository = new MemoryMemberRepository(); // 같은 리포지토리를 사용하도록 설정
    }

    @AfterEach
    public void afterEach() {
        memoryMemberRepository.clearStore();
    } // Shift + F10 을 통해 이전에 실행했던 동작 실행

    @Test
    void join() {
        //given -> 어떠한 상황에서
        Member member = new Member();
        member.setName("hello");

        //when -> 실행했을 때
        Long saveId = memberService.join(member);


        //then -> 이러한 결과가 나와야한다
        Member findMember = memberService.findOne(saveId).get();

        Assertions.assertThat(member.getName()).isEqualTo(findMember.getName());
    }

    @Test
    public void 중복_회원(){
        //given
        Member member1 = new Member();
        member1.setName("spring");

        Member member2 = new Member();
        member2.setName("spring");

        //when
        memberService.join(member1);
        IllegalStateException e = assertThrows(IllegalStateException.class, () -> memberService.join(member2));
        // 위 Exception이 발생해야 해, -> 다음 로직이 발생했을 때
        Assertions.assertThat(e.getMessage()).isEqualTo("이미 존재하는 회원입니다."); // 문자 비교
    }

    @Test
    void findMembers() {
    }

    @Test
    void findOne() {
    }
}