package TUK_Graduation_Work.GW_backend.repository;

import TUK_Graduation_Work.GW_backend.domain.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface UserRepository extends JpaRepository<User, Long> {

    // name으로 검색하는 메서드
    Optional<User> findByName(String name);
}
