"""
安全性模組

包含認證、授權、密碼哈希、JWT 處理和安全標頭等功能
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, status, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from jose import JWTError, jwt
import secrets
import hashlib
import hmac

from app.core.config import settings


class SecurityHeaders:
    """安全標頭中間件"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, request: Request, call_next):
        response = await call_next(request)

        # 添加安全標頭
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
                "font-src 'self'; "
                "connect-src 'self'; "
                "frame-ancestors 'none';"
            ),
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": (
                "geolocation=(), "
                "microphone=(), "
                "camera=(), "
                "payment=(), "
                "usb=(), "
                "magnetometer=(), "
                "gyroscope=(), "
                "accelerometer=()"
            )
        }

        for header, value in security_headers.items():
            response.headers[header] = value

        return response


# 密碼加密上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Bearer 認證
security = HTTPBearer(auto_error=False)


class PasswordManager:
    """密碼管理器"""

    @staticmethod
    def hash_password(password: str) -> str:
        """哈希密碼"""
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """驗證密碼"""
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def generate_secure_password(length: int = 12) -> str:
        """生成安全密碼"""
        return secrets.token_urlsafe(length)


class JWTManager:
    """JWT 管理器"""

    @staticmethod
    def create_access_token(
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """建立存取權杖"""
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
            )

        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        })

        return jwt.encode(
            to_encode,
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM
        )

    @staticmethod
    def create_refresh_token(data: Dict[str, Any]) -> str:
        """建立重新整理權杖"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        })

        return jwt.encode(
            to_encode,
            settings.SECRET_KEY,
            algorithm=settings.ALGORITHM
        )

    @staticmethod
    def decode_token(token: str) -> Dict[str, Any]:
        """解碼權杖"""
        try:
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.ALGORITHM]
            )
            return payload
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="無效的權杖",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @staticmethod
    def verify_token_type(payload: Dict[str, Any], expected_type: str) -> bool:
        """驗證權杖類型"""
        return payload.get("type") == expected_type


class DataEncryption:
    """資料加密器（用於敏感資料）"""

    @staticmethod
    def encrypt_data(data: str) -> str:
        """加密資料（簡化版，生產環境應使用更強的加密）"""
        key = settings.ENCRYPTION_KEY.encode()
        data_bytes = data.encode()

        # 使用 HMAC-SHA256 進行簡單加密（實際應用應使用 AES）
        signature = hmac.new(key, data_bytes, hashlib.sha256).hexdigest()
        return f"{signature}:{data}"

    @staticmethod
    def decrypt_data(encrypted_data: str) -> str:
        """解密資料"""
        try:
            signature, data = encrypted_data.split(":", 1)
            key = settings.ENCRYPTION_KEY.encode()
            data_bytes = data.encode()

            # 驗證簽名
            expected_signature = hmac.new(key, data_bytes, hashlib.sha256).hexdigest()
            if not hmac.compare_digest(signature, expected_signature):
                raise ValueError("資料完整性驗證失敗")

            return data
        except Exception:
            raise ValueError("解密失敗")


class SecurityValidator:
    """安全性驗證器"""

    @staticmethod
    def validate_password_strength(password: str) -> Dict[str, Any]:
        """驗證密碼強度"""
        issues = []
        score = 0

        # 長度檢查
        if len(password) >= 8:
            score += 1
        else:
            issues.append("密碼長度至少需要 8 個字符")

        # 複雜度檢查
        if any(c.islower() for c in password):
            score += 1
        else:
            issues.append("密碼需要包含小寫字母")

        if any(c.isupper() for c in password):
            score += 1
        else:
            issues.append("密碼需要包含大寫字母")

        if any(c.isdigit() for c in password):
            score += 1
        else:
            issues.append("密碼需要包含數字")

        if any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            score += 1
        else:
            issues.append("密碼需要包含特殊字符")

        strength_levels = ["很弱", "弱", "一般", "強", "很強"]
        strength = strength_levels[min(score, 4)]

        return {
            "score": score,
            "strength": strength,
            "is_valid": score >= 3,
            "issues": issues
        }

    @staticmethod
    def validate_file_upload(filename: str, content_type: str) -> bool:
        """驗證檔案上傳安全性"""
        # 檢查副檔名
        if not any(filename.lower().endswith(ext) for ext in settings.ALLOWED_EXTENSIONS):
            return False

        # 檢查 MIME 類型
        allowed_mime_types = {
            "image/jpeg",
            "image/jpg",
            "image/png"
        }
        if content_type not in allowed_mime_types:
            return False

        return True

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """清理檔名"""
        import re
        # 移除路徑分隔符和其他危險字符
        filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        # 限制長度
        if len(filename) > 255:
            name, ext = filename.rsplit('.', 1)
            filename = f"{name[:250]}.{ext}"
        return filename


class RateLimiter:
    """速率限制器"""

    def __init__(self):
        self._requests = {}

    def is_allowed(self, key: str, max_requests: int, window: int) -> bool:
        """檢查是否允許請求"""
        now = datetime.utcnow().timestamp()
        window_start = now - window

        # 清理過期記錄
        if key in self._requests:
            self._requests[key] = [
                req_time for req_time in self._requests[key]
                if req_time > window_start
            ]
        else:
            self._requests[key] = []

        # 檢查是否超過限制
        if len(self._requests[key]) >= max_requests:
            return False

        # 記錄當前請求
        self._requests[key].append(now)
        return True


# 全域實例
password_manager = PasswordManager()
jwt_manager = JWTManager()
data_encryption = DataEncryption()
security_validator = SecurityValidator()
rate_limiter = RateLimiter()