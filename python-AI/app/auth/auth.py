from fastapi_users import BaseUserManager, IntegerIDMixin
from sqlalchemy.orm import Session
from db_models.user import User

class UserManager(IntegerIDMixin, BaseUserManager[User, int]):
    reset_password_token_secret = "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6"
    verification_token_secret = "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6"

    async def on_after_register(self, user: User, request=None):
        print(f"User {user.id} has registered.")