package hello.hellospring.repository;

import hello.hellospring.domain.Member;
import org.springframework.stereotype.Repository;

import java.util.*;

// @Repository // 리포지토리를 컨테이너에 등록해줌
public class MemoryMemberRepository implements MemberRepository {

    /** * 동시성 문제가 고려되어 있지 않음, 실무에서는 ConcurrentHashMap, AtomicLong 사용 고려 */

    // Map을 통해 Member 저장, <데이터형식, 저장할 데이터> store =
    private static Map<Long, Member> store = new HashMap<>();
    private static long sequence = 0L; // key 값 생성

    @Override
    public Member save(Member member) {
        member.setId(++sequence); // id에 sequence 값 세팅
        store.put(member.getID(), member); // store에 저장하면 Map에 저장됨
        return member; // 결과 반환
    }

    @Override
    public Optional<Member> findById(Long id) {
        // store에서 찾은 아이디 반환 Optional을 통해 NULL 값도 감싸서 반환 후 클라이언트 동작
        return Optional.ofNullable(store.get(id));
    }

    @Override
    public Optional<Member> findByName(String name) {
        // lamda 함수를 통해 getName 을 통해 넘어온 데이터가 name과 같은 지 비교해 필터링 (루프) 만약 끝까지 없으면 Optional에 NULL로 반환
        return store.values().stream()
                .filter(member -> member.getName().equals(name))
                .findAny(); // 하나라도 찾는다면 findAny 의 결과까지 Optional로 반환

    }

    @Override
    public List<Member> findAll() {
        // 저장은 Map 이지만 반환은 List, loop 돌리기 편해 실무에 사용
        return new ArrayList<>(store.values()); // store.values()가 멤버들
    }

    public void clearStore(){
        store.clear();
    }
}
