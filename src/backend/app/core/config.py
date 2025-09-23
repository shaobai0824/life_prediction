"""
配置管理模組

使用 Pydantic Settings 管理環境變數和應用程式配置
"""

from typing import List, Optional
from pydantic import BaseSettings, validator
from decouple import config
import secrets


class Settings(BaseSettings):
    """應用程式設定"""

    # 基本設定
    APP_NAME: str = "壽命預測保險決策輔助系統"
    VERSION: str = "1.0.0"
    AUTHOR: str = "shaobai"
    ENVIRONMENT: str = config("ENVIRONMENT", default="development")
    DEBUG: bool = config("DEBUG", default=True, cast=bool)

    # 安全設定
    SECRET_KEY: str = config("SECRET_KEY", default=secrets.token_urlsafe(32))
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # 資料庫設定
    DATABASE_URL: str = config(
        "DATABASE_URL",
        default="sqlite+aiosqlite:///./life_prediction.db"
    )
    DATABASE_ECHO: bool = config("DATABASE_ECHO", default=False, cast=bool)

    # Redis 設定
    REDIS_URL: str = config("REDIS_URL", default="redis://localhost:6379/0")
    REDIS_EXPIRE_SECONDS: int = config("REDIS_EXPIRE_SECONDS", default=3600, cast=int)

    # CORS 設定
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",  # React 開發伺服器
        "http://localhost:5173",  # Vite 開發伺服器
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    # 信任主機
    ALLOWED_HOSTS: List[str] = [
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
    ]

    # 檔案上傳設定
    MAX_UPLOAD_SIZE: int = config("MAX_UPLOAD_SIZE", default=5 * 1024 * 1024, cast=int)  # 5MB
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png"]
    UPLOAD_TEMP_DIR: str = "./tmp/uploads"

    # AI 模型設定
    MODEL_PATH: str = config("MODEL_PATH", default="./models/trained/")
    MODEL_NAME: str = config("MODEL_NAME", default="life_prediction_model.pth")
    BATCH_SIZE: int = config("BATCH_SIZE", default=1, cast=int)
    DEVICE: str = config("DEVICE", default="cpu")  # cpu 或 cuda

    # 速率限制設定
    RATE_LIMIT_REQUESTS: int = config("RATE_LIMIT_REQUESTS", default=100, cast=int)
    RATE_LIMIT_WINDOW: int = config("RATE_LIMIT_WINDOW", default=3600, cast=int)  # 1小時

    # 日誌設定
    LOG_LEVEL: str = config("LOG_LEVEL", default="INFO")
    LOG_FILE: str = config("LOG_FILE", default="./logs/app.log")

    # 郵件設定 (用於通知和合規)
    SMTP_SERVER: Optional[str] = config("SMTP_SERVER", default=None)
    SMTP_PORT: int = config("SMTP_PORT", default=587, cast=int)
    SMTP_USERNAME: Optional[str] = config("SMTP_USERNAME", default=None)
    SMTP_PASSWORD: Optional[str] = config("SMTP_PASSWORD", default=None)
    EMAIL_FROM: str = config("EMAIL_FROM", default="noreply@lifepredict.com")

    # 監控設定
    PROMETHEUS_ENABLED: bool = config("PROMETHEUS_ENABLED", default=True, cast=bool)
    METRICS_PATH: str = "/metrics"

    # 金融合規設定
    AUDIT_LOG_ENABLED: bool = config("AUDIT_LOG_ENABLED", default=True, cast=bool)
    DATA_RETENTION_DAYS: int = config("DATA_RETENTION_DAYS", default=365, cast=int)
    ENCRYPTION_KEY: str = config("ENCRYPTION_KEY", default=secrets.token_urlsafe(32))

    @validator("CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v):
        """組裝 CORS 來源列表"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @validator("ALLOWED_HOSTS", pre=True)
    def assemble_allowed_hosts(cls, v):
        """組裝允許的主機列表"""
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v

    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        """驗證環境變數"""
        allowed_envs = {"development", "staging", "production"}
        if v not in allowed_envs:
            raise ValueError(f"ENVIRONMENT 必須是 {allowed_envs} 之一")
        return v

    @property
    def is_development(self) -> bool:
        """是否為開發環境"""
        return self.ENVIRONMENT == "development"

    @property
    def is_production(self) -> bool:
        """是否為生產環境"""
        return self.ENVIRONMENT == "production"

    @property
    def database_url_sync(self) -> str:
        """同步資料庫 URL（用於 Alembic）"""
        return self.DATABASE_URL.replace("+aiosqlite", "").replace("+asyncpg", "+psycopg2")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# 建立全域設定實例
settings = Settings()


# 開發環境專用設定
if settings.is_development:
    settings.CORS_ORIGINS.extend([
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ])
    settings.ALLOWED_HOSTS.extend([
        "testserver",
    ])


# 生產環境安全性檢查
if settings.is_production:
    assert settings.SECRET_KEY != "changeme", "生產環境必須設定安全的 SECRET_KEY"
    assert settings.ENCRYPTION_KEY != "changeme", "生產環境必須設定安全的 ENCRYPTION_KEY"
    assert not settings.DEBUG, "生產環境必須關閉 DEBUG 模式"