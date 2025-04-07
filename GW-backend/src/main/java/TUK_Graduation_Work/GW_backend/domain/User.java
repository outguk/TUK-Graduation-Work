package TUK_Graduation_Work.GW_backend.domain;

import jakarta.persistence.*;

@Entity
@Table(name="users")  // DB 테이블 이름
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;  // PK (AUTO_INCREMENT)

    @Column(nullable = false, unique = true)
    private String username;  // 사용자 이름(고유). 기존 "name" -> "username"

    @Column(nullable = false)
    private String password;  // 비밀번호(실제 운영시 해싱 필요)

    // 추가로 email / provider / socialId 등이 필요하다면 여기에 추가
    // private String email;
    // private String provider;
    // private String socialId;

    // ==== Getter/Setter ====
    public Long getId() {
        return id;
    }
    public void setId(Long id) {
        this.id = id;
    }

    public String getUsername() {
        return username;
    }
    public void setUsername(String username) {
        this.username = username;
    }

    public String getPassword() {
        return password;
    }
    public void setPassword(String password) {
        this.password = password;
    }
}