"""
用戶相關的 Pydantic 模型

定義 API 的請求和回應格式
"""

from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, Dict, Any
from datetime import datetime
import re


class UserBase(BaseModel):
    """用戶基礎資訊"""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50, description="用戶名")
    full_name: Optional[str] = Field(None, max_length=200, description="真實姓名")


class UserCreate(UserBase):
    """建立用戶請求"""
    password: str = Field(..., min_length=8, max_length=100, description="密碼")
    confirm_password: str = Field(..., description="確認密碼")

    # 隱私同意
    consent_data_processing: bool = Field(..., description="同意資料處理")
    consent_marketing: Optional[bool] = Field(False, description="同意行銷資訊")
    consent_analytics: Optional[bool] = Field(False, description="同意分析使用")

    # 可選資訊
    phone: Optional[str] = Field(None, description="手機號碼")
    preferred_language: Optional[str] = Field("zh-TW", description="偏好語言")

    @validator('password')
    def validate_password(cls, v):
        """驗證密碼強度"""
        if len(v) < 8:
            raise ValueError('密碼至少需要8個字符')

        # 檢查是否包含數字、字母和特殊字符
        if not re.search(r'\d', v):
            raise ValueError('密碼必須包含至少一個數字')
        if not re.search(r'[a-zA-Z]', v):
            raise ValueError('密碼必須包含至少一個字母')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('密碼必須包含至少一個特殊字符')

        return v

    @validator('confirm_password')
    def passwords_match(cls, v, values):
        """驗證密碼確認"""
        if 'password' in values and v != values['password']:
            raise ValueError('密碼確認不匹配')
        return v

    @validator('phone')
    def validate_phone(cls, v):
        """驗證手機號碼格式"""
        if v and not re.match(r'^(\+886|0)[0-9]{9}$', v):
            raise ValueError('請輸入有效的台灣手機號碼')
        return v

    @validator('consent_data_processing')
    def validate_consent(cls, v):
        """驗證必要同意"""
        if not v:
            raise ValueError('必須同意資料處理條款才能註冊')
        return v


class UserLogin(BaseModel):
    """用戶登入請求"""
    email: EmailStr = Field(..., description="電子郵件")
    password: str = Field(..., description="密碼")
    remember_me: Optional[bool] = Field(False, description="記住我")
    device_info: Optional[Dict[str, Any]] = Field(None, description="裝置資訊")


class UserResponse(BaseModel):
    """用戶資訊回應"""
    user_id: str
    email: EmailStr
    username: str
    full_name: Optional[str]
    is_active: bool
    is_verified: bool
    is_premium: bool
    created_at: Optional[datetime]
    last_login: Optional[datetime]
    preferred_language: str
    total_predictions: int
    remaining_predictions: int

    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    """更新用戶資訊請求"""
    full_name: Optional[str] = Field(None, max_length=200)
    phone: Optional[str] = Field(None)
    preferred_language: Optional[str] = Field(None)

    @validator('phone')
    def validate_phone(cls, v):
        if v and not re.match(r'^(\+886|0)[0-9]{9}$', v):
            raise ValueError('請輸入有效的台灣手機號碼')
        return v


class PasswordChange(BaseModel):
    """變更密碼請求"""
    current_password: str = Field(..., description="目前密碼")
    new_password: str = Field(..., min_length=8, description="新密碼")
    confirm_password: str = Field(..., description="確認新密碼")

    @validator('new_password')
    def validate_password(cls, v):
        """驗證密碼強度"""
        if len(v) < 8:
            raise ValueError('密碼至少需要8個字符')

        if not re.search(r'\d', v):
            raise ValueError('密碼必須包含至少一個數字')
        if not re.search(r'[a-zA-Z]', v):
            raise ValueError('密碼必須包含至少一個字母')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('密碼必須包含至少一個特殊字符')

        return v

    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('密碼確認不匹配')
        return v


class PasswordReset(BaseModel):
    """密碼重設請求"""
    email: EmailStr = Field(..., description="註冊時的電子郵件")


class PasswordResetConfirm(BaseModel):
    """確認密碼重設"""
    token: str = Field(..., description="重設令牌")
    verification_code: str = Field(..., description="驗證碼")
    new_password: str = Field(..., min_length=8, description="新密碼")
    confirm_password: str = Field(..., description="確認新密碼")

    @validator('new_password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('密碼至少需要8個字符')

        if not re.search(r'\d', v):
            raise ValueError('密碼必須包含至少一個數字')
        if not re.search(r'[a-zA-Z]', v):
            raise ValueError('密碼必須包含至少一個字母')
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', v):
            raise ValueError('密碼必須包含至少一個特殊字符')

        return v

    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('密碼確認不匹配')
        return v


class EmailVerification(BaseModel):
    """電子郵件驗證"""
    token: str = Field(..., description="驗證令牌")
    verification_code: str = Field(..., description="驗證碼")


class LoginResponse(BaseModel):
    """登入成功回應"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class TokenRefresh(BaseModel):
    """刷新令牌請求"""
    refresh_token: str = Field(..., description="刷新令牌")


class UserPreferencesUpdate(BaseModel):
    """用戶偏好設定更新"""
    email_notifications: Optional[bool] = None
    sms_notifications: Optional[bool] = None
    push_notifications: Optional[bool] = None
    theme: Optional[str] = Field(None, regex="^(light|dark|fortune)$")
    language: Optional[str] = Field(None, regex="^(zh-TW|zh-CN|en)$")
    profile_visibility: Optional[str] = Field(None, regex="^(private|friends|public)$")
    data_retention_days: Optional[int] = Field(None, ge=30, le=365)
    auto_save_images: Optional[bool] = None
    detailed_reports: Optional[bool] = None


class UserPreferencesResponse(BaseModel):
    """用戶偏好設定回應"""
    email_notifications: bool
    sms_notifications: bool
    push_notifications: bool
    theme: str
    language: str
    profile_visibility: str
    data_retention_days: int
    auto_save_images: bool
    detailed_reports: bool

    class Config:
        from_attributes = True


class UserStats(BaseModel):
    """用戶統計資訊"""
    total_predictions: int
    remaining_predictions: int
    this_month_predictions: int
    accuracy_score: Optional[float]
    member_since: datetime
    last_prediction: Optional[datetime]

    class Config:
        from_attributes = True