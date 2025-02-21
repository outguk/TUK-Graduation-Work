package TUK_Graduation_Work.GW_backend.service;

import TUK_Graduation_Work.GW_backend.domain.User;
import TUK_Graduation_Work.GW_backend.repository.UserRepository;
import TUK_Graduation_Work.GW_backend.repository.MemoryUserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.Optional;

public class UserService {
    private final UserRepository userRepository;

    public UserService(UserRepository userRepository){
        this.userRepository = userRepository;
    }

    // 회원 가입
    public String join(User user){
        // 같은 이름 중복 회원 금지
        validateDuplicateMember(user); //

        userRepository.save(user);
        return user.getName();
    }

    private void validateDuplicateMember(User user) {
        userRepository.findByName(user.getName())
                .ifPresent(m -> {
                    throw new IllegalStateException("이미 존재하는 회원입니다.");
                });
    }

    // 전체 회원 조회
    public List<User> findUsers(){
        return userRepository.findAll();
    }

    public Optional<User> findOne(String userName) {
        return userRepository.findByName(userName);
    }

}
