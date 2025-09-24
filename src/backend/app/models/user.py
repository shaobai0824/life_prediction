"""
用戶資料模型

處理用戶帳號、認證和個人資訊
"""

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime
import uuid

from ..core.database import Base
from ..core.security import get_password_hash, verify_password

Base = declarative_base()


class User(Base):
    """用戶模型"""

    __tablename__ = "users"

    # 基本資訊
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)

    # 個人資訊
    full_name = Column(String(200), nullable=True)
    phone = Column(String(20), nullable=True)
    date_of_birth = Column(DateTime, nullable=True)

    # 帳號狀態
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    is_premium = Column(Boolean, default=False)

    # 系統欄位
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)

    # 隱私設定
    consent_data_processing = Column(Boolean, default=False)
    consent_marketing = Column(Boolean, default=False)
    consent_analytics = Column(Boolean, default=False)

    # 用戶偏好
    preferred_language = Column(String(10), default="zh-TW")
    timezone = Column(String(50), default="Asia/Taipei")

    # 預測記錄
    total_predictions = Column(Integer, default=0)
    remaining_predictions = Column(Integer, default=3)  # 免費用戶限制

    def set_password(self, password: str):
        """設定密碼"""
        self.hashed_password = get_password_hash(password)

    def verify_password(self, password: str) -> bool:
        """驗證密碼"""
        return verify_password(password, self.hashed_password)

    def to_dict(self) -> dict:
        """轉換為字典格式"""
        return {
            'user_id': self.user_id,
            'email': self.email,
            'username': self.username,
            'full_name': self.full_name,
            'is_active': self.is_active,
            'is_verified': self.is_verified,
            'is_premium': self.is_premium,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'preferred_language': self.preferred_language,
            'total_predictions': self.total_predictions,
            'remaining_predictions': self.remaining_predictions
        }

    def can_predict(self) -> bool:
        """檢查是否可以進行預測"""
        if self.is_premium:
            return True
        return self.remaining_predictions > 0

    def use_prediction(self):
        """使用一次預測"""
        if not self.can_predict():
            raise ValueError("預測次數已用完")

        self.total_predictions += 1
        if not self.is_premium:
            self.remaining_predictions -= 1


class UserSession(Base):
    """用戶會話模型"""

    __tablename__ = "user_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, index=True, nullable=False)
    user_id = Column(String(36), index=True, nullable=False)

    # 會話資訊
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    device_type = Column(String(50), nullable=True)

    # 時間戳記
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)
    last_activity = Column(DateTime(timezone=True), server_default=func.now())

    # 狀態
    is_active = Column(Boolean, default=True)
    logout_at = Column(DateTime(timezone=True), nullable=True)

    def is_expired(self) -> bool:
        """檢查會話是否過期"""
        return datetime.utcnow() > self.expires_at

    def is_valid(self) -> bool:
        """檢查會話是否有效"""
        return self.is_active and not self.is_expired()


class UserVerification(Base):
    """用戶驗證模型"""

    __tablename__ = "user_verifications"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(36), index=True, nullable=False)

    # 驗證資訊
    verification_type = Column(String(50), nullable=False)  # email, phone, reset_password
    verification_code = Column(String(10), nullable=False)
    verification_token = Column(String(255), unique=True, nullable=False)

    # 狀態
    is_used = Column(Boolean, default=False)
    attempts = Column(Integer, default=0)
    max_attempts = Column(Integer, default=3)

    # 時間戳記
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)
    used_at = Column(DateTime(timezone=True), nullable=True)

    def is_expired(self) -> bool:
        """檢查驗證碼是否過期"""
        return datetime.utcnow() > self.expires_at

    def is_valid(self) -> bool:
        """檢查驗證碼是否有效"""
        return not self.is_used and not self.is_expired() and self.attempts < self.max_attempts

    def use_verification(self):
        """使用驗證碼"""
        if not self.is_valid():
            raise ValueError("驗證碼無效或已過期")

        self.is_used = True
        self.used_at = datetime.utcnow()


class UserPreferences(Base):
    """用戶偏好設定模型"""

    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(36), unique=True, index=True, nullable=False)

    # 通知設定
    email_notifications = Column(Boolean, default=True)
    sms_notifications = Column(Boolean, default=False)
    push_notifications = Column(Boolean, default=True)

    # UI 偏好
    theme = Column(String(20), default="light")  # light, dark, fortune
    language = Column(String(10), default="zh-TW")

    # 隱私設定
    profile_visibility = Column(String(20), default="private")  # private, friends, public
    data_retention_days = Column(Integer, default=90)

    # 功能設定
    auto_save_images = Column(Boolean, default=False)
    detailed_reports = Column(Boolean, default=True)

    # 更新時間
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def to_dict(self) -> dict:
        """轉換為字典格式"""
        return {
            'email_notifications': self.email_notifications,
            'sms_notifications': self.sms_notifications,
            'push_notifications': self.push_notifications,
            'theme': self.theme,
            'language': self.language,
            'profile_visibility': self.profile_visibility,
            'data_retention_days': self.data_retention_days,
            'auto_save_images': self.auto_save_images,
            'detailed_reports': self.detailed_reports
        }