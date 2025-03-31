package TUK_Graduation_Work.GW_backend;

import TUK_Graduation_Work.GW_backend.repository.UserRepository;
import TUK_Graduation_Work.GW_backend.service.UserService;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.core.io.ClassPathResource;
import org.springframework.http.MediaType;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.server.*;

@Configuration
public class SpringConfig {

    // 만약 UserService에 대한 Bean 등록을 수동으로 하려면:
    @Bean
    public UserService userService(UserRepository userRepository) {
        return new UserService(userRepository);
    }

    // WebClient
    @Bean
    public WebClient webClient() {
        return WebClient.builder()
                .baseUrl("https://jsonplaceholder.typicode.com")
                .build();
    }

    // SPA 라우팅 Fallback
    @Bean
    public RouterFunction<ServerResponse> spaRouter() {
        return RouterFunctions
                .resources("/**", new ClassPathResource("static/"))
                .andRoute(RequestPredicates.GET("/**"), request ->
                        ServerResponse.ok()
                                .contentType(MediaType.TEXT_HTML)
                                .body(BodyInserters.fromResource(new ClassPathResource("static/index.html")))
                );
    }
}
