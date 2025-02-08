package hello.hellospring.service;

import hello.hellospring.domain.Member;
import hello.hellospring.repository.MemberRepository;
import hello.hellospring.repository.MemoryMemberRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

// @Service // 컨테이너에 서비스를 등록해줌
public class MemberService { // test 쉽게 만들기 Ctrl + Shift + T

    private final MemberRepository memberRepository;

    // 리포지토리를 공유하도록 설정
    // @Autowired // MemberService가 memberRepository를 필요로 하므로 연결 (test case 에서도 new를 이용하면 아예 새로운 객체가 생성됨)
    public MemberService(MemberRepository memberRepository){ // 외부에서  memberRepository를 가져오도록 설정
        this.memberRepository = memberRepository;
    }

    // 회원 가입
    public Long join(Member member){
        // 같은 이름 중복 회원 금지
        validateDuplicateMember(member); //

        memberRepository.save(member);
        return member.getID();
    }

    private void validateDuplicateMember(Member member) {
        memberRepository.findByName(member.getName()) // Ctrl + Alt + v -> 반환값 자동 설정, Ctrl + Alt + m -> 메서드로 만들기
            .ifPresent(m -> {
                throw new IllegalStateException("이미 존재하는 회원입니다.");
            });
    }

    // 전체 회원 조회
    public List<Member> findMembers(){
        return memberRepository.findAll();
    }

    public Optional<Member> findOne(Long memberId) {
        return memberRepository.findById(memberId);
    }
}
