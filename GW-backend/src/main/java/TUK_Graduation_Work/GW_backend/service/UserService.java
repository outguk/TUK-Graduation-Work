package TUK_Graduation_Work.GW_backend.service;

import TUK_Graduation_Work.GW_backend.domain.User;
import TUK_Graduation_Work.GW_backend.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Optional;

@Service
public class UserService {

    private final UserRepository userRepository;

    @Autowired
    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }

    // 회원가입 로직
    public void join(User user) {
        // 예: 중복 이름 체크, 암호화 등
        userRepository.save(user);
    }

    // 사용자 조회
    public Optional<User> findOne(String name) {
        return userRepository.findByName(name);
    }
}