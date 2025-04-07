package TUK_Graduation_Work.GW_backend.repository;

import TUK_Graduation_Work.GW_backend.domain.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface UserRepository extends JpaRepository<User, Long> {
    // username으로 사용자 조회
    Optional<User> findByUsername(String username);
}