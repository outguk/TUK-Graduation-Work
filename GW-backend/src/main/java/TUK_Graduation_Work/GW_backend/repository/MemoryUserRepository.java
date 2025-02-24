package TUK_Graduation_Work.GW_backend.repository;

import java.util.*;
import TUK_Graduation_Work.GW_backend.domain.User;
import org.springframework.stereotype.Repository;


public class MemoryUserRepository implements UserRepository {

    /** * 동시성 문제가 고려되어 있지 않음, 실무에서는 ConcurrentHashMap, AtomicLong 사용 고려 */

    private static Map<String, User> store = new HashMap<>();

    @Override
    public User save(User user) {
        store.put(user.getName(), user);
        return user;
    }

    @Override
    public Optional<User> findByName(String name) {
        return store.values().stream()
                .filter(user -> user.getName().equals(name))
                .findAny();
    }

    @Override
    public List<User> findAll() {
        return new ArrayList<>(store.values());
    }

    public void clearStore(){
        store.clear();
    }
}
