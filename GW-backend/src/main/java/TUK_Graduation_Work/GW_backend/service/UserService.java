package TUK_Graduation_Work.GW_backend.service;

import TUK_Graduation_Work.GW_backend.domain.User;
import TUK_Graduation_Work.GW_backend.repository.UserRepository;
import reactor.core.publisher.Mono;

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

    // 회원가입 (블로킹 -> Mono)
    public Mono<User> joinAsync(User user) {
        return Mono.fromCallable(() -> {
            // 중복체크, 비번 해싱 등
            return userRepository.save(user);
        });
    }

    // 사용자조회 (username) (블로킹 -> Mono)
    public Mono<User> findOneAsync(String username) {
        return Mono.fromCallable(() -> 
            userRepository.findByUsername(username).orElse(null)
        );
    }
}