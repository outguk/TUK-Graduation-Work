package TUK_Graduation_Work.GW_backend.repository;
import TUK_Graduation_Work.GW_backend.domain.User;

import java.util.List;
import java.util.Optional;

public interface UserRepository {
    User save(User member);
    Optional<User> findByName(String name);
    List<User> findAll();
}
