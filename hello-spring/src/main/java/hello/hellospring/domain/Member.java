package hello.hellospring.domain;

public class Member {

    private Long id;
    private String name;

    public Long getID() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public String getName() {
        return name;
    }

    public void setName(String name){
        this.name=name;
    }
    // Alt + Insert 를 통해 Constructor 단축키 -> getter setter
}
