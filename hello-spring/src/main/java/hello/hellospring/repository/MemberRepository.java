package hello.hellospring.repository;

import hello.hellospring.domain.Member;

import java.util.List;
import java.util.Optional;

public interface MemberRepository {
    Member save(Member member); // 회원 저장시 저장된 회원 반환
    // Optional -> java8에 들어간 NULL을 처리하는 방법 (Optional을 통해 NULL을 감싸서 반환함)
    Optional<Member> findById(Long id); // id로 회원 찾기
    Optional<Member> findByName(String name); // 이름으로 회원 찾기
    List<Member> findAll(); // 지금까지 저장된 회원 리스트를 반환

}
