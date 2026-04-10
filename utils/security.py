import hashlib

# 使用 SHA256 进行密码哈希（简单但足够安全）
def get_hashed_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password
    