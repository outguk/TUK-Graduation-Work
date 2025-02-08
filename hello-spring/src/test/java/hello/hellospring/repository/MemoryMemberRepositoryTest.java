package hello.hellospring.repository;

import hello.hellospring.domain.Member;
import org.assertj.core.api.Assertions;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import java.awt.*;
import java.util.List;

import static org.assertj.core.api.Assertions.*;

class MemoryMemberRepositoryTest {
    MemoryMemberRepository repository = new MemoryMemberRepository(); // Ctrl + 클릭 을 통해 해당 파일로 바로 갈 수 있음

    @AfterEach
    public void afterEach() {
        repository.clearStore();
    }

    @Test
    public void save() {
        Member member = new Member(); // test용 member 생성
        member.setName("spring"); // name 설정

        repository.save(member); // 실제 저장해보기

        /* getId는 반환 타입이 Optional 이다. 이때 get을 이용해 Optional에서 꺼낼 수 있다.*/
        Member result = repository.findById(member.getID()).get(); // name이 저장되어있는지 확인
        assertThat(member).isEqualTo(result); // Assertions 사용 시 org.assertj 사용 -> 더 간결하게 만들어줌
    }

    @Test
    public void findByName(){
        Member member1 = new Member();
        member1.setName("spring1");
        repository.save(member1);


        Member member2 = new Member(); // shift + F6 을 통해 변수 rename 가능
        member2.setName("spring2");
        repository.save(member2);

        Member result = repository.findByName("spring1").get();
        assertThat(result).isEqualTo(member1);

    }

    @Test
    public void findAll(){
        Member member1 = new Member();
        member1.setName("spring1");
        repository.save(member1);

        Member member2 = new Member();
        member2.setName("spring2");
        repository.save(member2);

        List<Member> result = repository.findAll();

        assertThat(result.size()).isEqualTo(2);
    }


}
