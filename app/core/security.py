from datetime import datetime, timedelta
from jose import JWTError, jwt
from app.core.config import settings
import hashlib

def verify_password(plain_password, hashed_password):
    # Temporal: SHA256 simple
    test_hash = hashlib.sha256(plain_password.encode()).hexdigest()
    return test_hash == hashed_password

def get_password_hash(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def verify_token(token: str):
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        return None