from pydantic_settings import SettingsConfigDict, BaseSettings


class Settings(BaseSettings):
    DB_HOST: str
    DB_PORT: int
    DB_USER: str
    DB_PASS: str
    DB_NAME: str
    
    POSTGRES_PASSWORD: str
    POSTGRES_USER: str
    POSTGRES_DB: str
    
    REDIS_HOST: str
    REDIS_PORT: int
    
    SECRET_KEY: str
    ALGORITHM: str
    
    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()