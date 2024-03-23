from functools import lru_cache
from pydantic_settings import BaseSettings

with open('.TOKEN', 'r') as f:
    app_auth_token = f.readline()

class Settings(BaseSettings):
    debug: bool = False
    echo_active: bool = False
    app_auth_token: str = app_auth_token
    skip_auth: bool = False

@lru_cache   
def get_settings():
    return Settings()


settings = Settings()