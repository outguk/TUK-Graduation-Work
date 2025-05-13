package TUK_Graduation_Work.GW_backend;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.core.io.ClassPathResource;
import org.springframework.http.MediaType;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.server.*;

@Configuration
public class SpringConfig {

    // Spring Data JPA를 사용하면, UserRepository와 UserService는
    // 각각 @Repository와 @Service 어노테이션으로 자동 등록되므로 수동 Bean 등록은 필요 없습니다.
    // 만약 수동 등록을 원한다면 아래와 같이 주입할 수 있습니다.
    /*
    @Bean
    public UserService userService(UserRepository userRepository) {
        return new UserService(userRepository);
    }
    */

    // WebClient 설정 (필요에 따라 baseUrl 수정)
    @Bean
    public WebClient webClient() {
        return WebClient.builder()
                .baseUrl("https://jsonplaceholder.typicode.com")
                .build();
    }

    // SPA 라우팅 Fallback
    @Bean
    public RouterFunction<ServerResponse> spaRouter() {

        RequestPredicate isApi = RequestPredicates.path("/spring/api/**");
        RequestPredicate isGet = RequestPredicates.GET("/**");

        return RouterFunctions
                .resources("/**", new ClassPathResource("static/"))
                // ①  GET 이고 ②  /spring/api/** 가 아닌 것만 index.html
                .andRoute(isGet.and(isApi.negate()), req ->
                        ServerResponse.ok()
                                .contentType(MediaType.TEXT_HTML)
                                .body(BodyInserters.fromResource(
                                        new ClassPathResource("static/index.html")))
                );
    }
}