package TUK_Graduation_Work.GW_backend;

import TUK_Graduation_Work.GW_backend.repository.MemoryUserRepository;
import TUK_Graduation_Work.GW_backend.repository.UserRepository;
import TUK_Graduation_Work.GW_backend.service.UserService;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.WebClient;

@Configuration
public class SpringConfig {

    @Bean
    public UserService userService(){
        return new UserService(userRepository());
    }

    @Bean
    public UserRepository userRepository(){
        return new MemoryUserRepository();
    }

    @Bean
    public WebClient webClient() {
        return WebClient.builder()
                .baseUrl("https://jsonplaceholder.typicode.com")
                .build();
    }
}
