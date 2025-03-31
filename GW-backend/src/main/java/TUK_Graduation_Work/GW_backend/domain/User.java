package TUK_Graduation_Work.GW_backend.domain;

import jakarta.persistence.*;

@Entity
@Table(name="users")  // 테이블명
public class User {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;  // PK (자동증가)

    private String name;     // 유저명
    private String password; // 암호
    
    // Getter/Setter ...
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public String getPassword() { return password; }
    public void setPassword(String password) { this.password = password; }
}