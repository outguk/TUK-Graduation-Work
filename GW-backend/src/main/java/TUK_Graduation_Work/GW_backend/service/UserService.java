package TUK_Graduation_Work.GW_backend.service;

import TUK_Graduation_Work.GW_backend.domain.User;
import TUK_Graduation_Work.GW_backend.repository.UserRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

@Service
public class UserService {
    private final UserRepository userRepository;
    private final BCryptPasswordEncoder passwordEncoder;

    @Autowired
    public UserService(UserRepository userRepository, BCryptPasswordEncoder passwordEncoder) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
    }

    public Mono<User> joinAsync(User user) {
        return Mono.fromCallable(() -> {
            if (userRepository.findByUsername(user.getUsername()).isPresent()) {
                throw new IllegalArgumentException("Username already exists");
            }
            if (user.getEmail() != null && userRepository.findByEmail(user.getEmail()).isPresent()) {
                throw new IllegalArgumentException("Email already exists");
            }
            user.setPassword(passwordEncoder.encode(user.getPassword()));
            return userRepository.save(user);
        });
    }

    public Mono<User> findOneAsync(String username) {
        return Mono.fromCallable(() -> 
            userRepository.findByUsername(username).orElse(null)
        );
    }

    public Mono<User> findByEmailAsync(String email) {
        return Mono.fromCallable(() -> 
            userRepository.findByEmail(email).orElse(null)
        );
    }

    public Mono<User> findByIdAsync(Long id) {
        return Mono.fromCallable(() -> 
            userRepository.findById(id).orElse(null)
        );
    }

    public Mono<Boolean> checkPassword(Long userId, String rawPassword) {
        return findByIdAsync(userId)
            .map(user -> passwordEncoder.matches(rawPassword, user.getPassword()));
    }

    public Mono<User> updateUserAsync(Long id, String name, String email, String newPassword) {
        return findByIdAsync(id)
            .switchIfEmpty(Mono.error(new IllegalArgumentException("User not found")))
            .flatMap(user -> {
                // 이름 업데이트
                if (name != null && !name.isEmpty()) {
                    user.setUsername(name);
                }

                // 이메일 업데이트
                if (email != null && !email.isEmpty()) {
                    return findByEmailAsync(email)
                        .flatMap(existingUser -> {
                            if (existingUser != null && !existingUser.getId().equals(id)) {
                                return Mono.error(new IllegalArgumentException("Email already in use"));
                            }
                            user.setEmail(email);
                            return saveUser(user, newPassword);
                        })
                        .switchIfEmpty(Mono.fromCallable(() -> {
                            user.setEmail(email);
                            return user;
                        }).flatMap(updatedUser -> saveUser(updatedUser, newPassword)));
                } else {
                    user.setEmail(null); // 클라이언트가 빈 이메일 보낼 경우 null로 설정
                    return saveUser(user, newPassword);
                }
            });
    }

    private Mono<User> saveUser(User user, String newPassword) {
        return Mono.fromCallable(() -> {
            if (newPassword != null && !newPassword.isEmpty()) {
                user.setPassword(passwordEncoder.encode(newPassword));
            }
            User savedUser = userRepository.save(user);
            System.out.println("Saved user - ID: " + savedUser.getId() + ", Username: " + savedUser.getUsername() + ", Email: " + savedUser.getEmail());
            return savedUser;
        });
    }
}