"""
認證相關 API 路由

處理用戶註冊、登入、驗證等功能
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import secrets
import logging

from ..core.database import get_db
from ..core.security import (
    create_access_token,
    create_refresh_token,
    verify_token,
    get_current_user,
    create_verification_token,
    verify_verification_token
)
from ..schemas.user import (
    UserCreate,
    UserLogin,
    UserResponse,
    LoginResponse,
    TokenRefresh,
    PasswordReset,
    PasswordResetConfirm,
    EmailVerification,
    PasswordChange,
    UserUpdate,
    UserPreferencesUpdate,
    UserPreferencesResponse
)
from ..models.user import User, UserSession, UserVerification, UserPreferences
from ..services.email import send_verification_email, send_password_reset_email
from ..services.user_service import UserService
from ..middleware.rate_limit import RateLimitConfig, rate_limit

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["認證"])
security = HTTPBearer()

# 速率限制配置
login_rate_limit = RateLimitConfig(max_requests=5, time_window=900)  # 15分鐘5次
register_rate_limit = RateLimitConfig(max_requests=3, time_window=3600)  # 1小時3次
password_reset_rate_limit = RateLimitConfig(max_requests=2, time_window=3600)  # 1小時2次


@router.post("/register", response_model=Dict[str, Any])
@rate_limit(register_rate_limit)
async def register_user(
    user_create: UserCreate,
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """用戶註冊"""
    try:
        user_service = UserService(db)

        # 檢查用戶是否已存在
        if user_service.get_user_by_email(user_create.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="此電子郵件已被註冊"
            )

        if user_service.get_user_by_username(user_create.username):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="此用戶名已被使用"
            )

        # 建立新用戶
        new_user = user_service.create_user(user_create)

        # 建立電子郵件驗證記錄
        verification_code = secrets.randbelow(900000) + 100000  # 6位數驗證碼
        verification_token = create_verification_token({
            "user_id": new_user.user_id,
            "email": new_user.email,
            "type": "email_verification"
        })

        verification = UserVerification(
            user_id=new_user.user_id,
            verification_type="email",
            verification_code=str(verification_code),
            verification_token=verification_token,
            expires_at=datetime.utcnow() + timedelta(hours=24)
        )
        db.add(verification)

        # 建立用戶偏好設定
        preferences = UserPreferences(
            user_id=new_user.user_id,
            language=user_create.preferred_language or "zh-TW"
        )
        db.add(preferences)

        db.commit()

        # 發送驗證郵件
        background_tasks.add_task(
            send_verification_email,
            new_user.email,
            new_user.username,
            verification_code,
            verification_token
        )

        logger.info(f"新用戶註冊成功: {new_user.email}")

        return {
            "success": True,
            "message": "註冊成功！請檢查您的電子郵件以完成驗證",
            "user_id": new_user.user_id,
            "verification_required": True
        }

    except IntegrityError as e:
        db.rollback()
        logger.error(f"註冊失敗 - 資料庫完整性錯誤: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="註冊失敗，請檢查輸入資料"
        )
    except Exception as e:
        db.rollback()
        logger.error(f"註冊失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="註冊失敗，請稍後再試"
        )


@router.post("/login", response_model=LoginResponse)
@rate_limit(login_rate_limit)
async def login_user(
    user_login: UserLogin,
    request: Request,
    db: Session = Depends(get_db)
):
    """用戶登入"""
    try:
        user_service = UserService(db)

        # 驗證用戶
        user = user_service.authenticate_user(user_login.email, user_login.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="電子郵件或密碼錯誤"
            )

        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="帳號已被停用"
            )

        # 更新最後登入時間
        user.last_login = datetime.utcnow()

        # 建立 tokens
        token_expires = timedelta(hours=24 if user_login.remember_me else 1)
        access_token = create_access_token(
            data={"sub": user.user_id, "email": user.email},
            expires_delta=token_expires
        )

        refresh_token = create_refresh_token(
            data={"sub": user.user_id, "email": user.email}
        )

        # 建立用戶會話
        session = UserSession(
            session_id=secrets.token_urlsafe(32),
            user_id=user.user_id,
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent"),
            device_type=user_login.device_info.get("type") if user_login.device_info else "web",
            expires_at=datetime.utcnow() + token_expires
        )
        db.add(session)
        db.commit()

        logger.info(f"用戶登入成功: {user.email}")

        return LoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=int(token_expires.total_seconds()),
            user=UserResponse.from_orm(user)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"登入失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="登入失敗，請稍後再試"
        )


@router.post("/refresh", response_model=Dict[str, Any])
async def refresh_token(
    token_refresh: TokenRefresh,
    db: Session = Depends(get_db)
):
    """刷新訪問令牌"""
    try:
        # 驗證 refresh token
        payload = verify_token(token_refresh.refresh_token, token_type="refresh")
        user_id = payload.get("sub")

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="無效的刷新令牌"
            )

        # 獲取用戶
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="用戶不存在或已被停用"
            )

        # 建立新的 access token
        access_token = create_access_token(
            data={"sub": user.user_id, "email": user.email}
        )

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": 3600  # 1小時
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"令牌刷新失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="令牌刷新失敗"
        )


@router.post("/verify-email", response_model=Dict[str, Any])
async def verify_email(
    verification: EmailVerification,
    db: Session = Depends(get_db)
):
    """驗證電子郵件"""
    try:
        # 驗證令牌
        payload = verify_verification_token(verification.token)
        user_id = payload.get("user_id")

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="無效的驗證令牌"
            )

        # 查找驗證記錄
        verification_record = db.query(UserVerification).filter(
            UserVerification.user_id == user_id,
            UserVerification.verification_token == verification.token,
            UserVerification.verification_code == verification.verification_code,
            UserVerification.verification_type == "email"
        ).first()

        if not verification_record or not verification_record.is_valid():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="驗證碼無效或已過期"
            )

        # 更新用戶狀態
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用戶不存在"
            )

        user.is_verified = True
        verification_record.use_verification()

        db.commit()

        logger.info(f"電子郵件驗證成功: {user.email}")

        return {
            "success": True,
            "message": "電子郵件驗證成功！您現在可以使用所有功能"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"電子郵件驗證失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="驗證失敗，請稍後再試"
        )


@router.post("/password-reset", response_model=Dict[str, Any])
@rate_limit(password_reset_rate_limit)
async def request_password_reset(
    password_reset: PasswordReset,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """請求密碼重設"""
    try:
        user = db.query(User).filter(User.email == password_reset.email).first()

        # 無論用戶是否存在都返回成功（安全考量）
        if user:
            # 建立密碼重設記錄
            verification_code = secrets.randbelow(900000) + 100000
            verification_token = create_verification_token({
                "user_id": user.user_id,
                "email": user.email,
                "type": "password_reset"
            })

            verification = UserVerification(
                user_id=user.user_id,
                verification_type="reset_password",
                verification_code=str(verification_code),
                verification_token=verification_token,
                expires_at=datetime.utcnow() + timedelta(hours=1)  # 1小時有效
            )
            db.add(verification)
            db.commit()

            # 發送重設郵件
            background_tasks.add_task(
                send_password_reset_email,
                user.email,
                user.username,
                verification_code,
                verification_token
            )

            logger.info(f"密碼重設請求: {user.email}")

        return {
            "success": True,
            "message": "如果該電子郵件已註冊，您將收到密碼重設指示"
        }

    except Exception as e:
        logger.error(f"密碼重設請求失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="請求失敗，請稍後再試"
        )


@router.post("/password-reset-confirm", response_model=Dict[str, Any])
async def confirm_password_reset(
    reset_confirm: PasswordResetConfirm,
    db: Session = Depends(get_db)
):
    """確認密碼重設"""
    try:
        # 驗證令牌
        payload = verify_verification_token(reset_confirm.token)
        user_id = payload.get("user_id")

        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="無效的重設令牌"
            )

        # 查找驗證記錄
        verification_record = db.query(UserVerification).filter(
            UserVerification.user_id == user_id,
            UserVerification.verification_token == reset_confirm.token,
            UserVerification.verification_code == reset_confirm.verification_code,
            UserVerification.verification_type == "reset_password"
        ).first()

        if not verification_record or not verification_record.is_valid():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="驗證碼無效或已過期"
            )

        # 更新密碼
        user = db.query(User).filter(User.user_id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用戶不存在"
            )

        user.set_password(reset_confirm.new_password)
        verification_record.use_verification()

        # 停用所有現有會話
        db.query(UserSession).filter(
            UserSession.user_id == user_id,
            UserSession.is_active == True
        ).update({"is_active": False, "logout_at": datetime.utcnow()})

        db.commit()

        logger.info(f"密碼重設成功: {user.email}")

        return {
            "success": True,
            "message": "密碼重設成功！請使用新密碼登入"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"密碼重設確認失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="密碼重設失敗，請稍後再試"
        )


@router.post("/logout", response_model=Dict[str, Any])
async def logout_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """用戶登出"""
    try:
        # 停用當前用戶的所有會話
        db.query(UserSession).filter(
            UserSession.user_id == current_user.user_id,
            UserSession.is_active == True
        ).update({"is_active": False, "logout_at": datetime.utcnow()})

        db.commit()

        logger.info(f"用戶登出: {current_user.email}")

        return {
            "success": True,
            "message": "登出成功"
        }

    except Exception as e:
        logger.error(f"登出失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="登出失敗，請稍後再試"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """獲取當前用戶資訊"""
    return UserResponse.from_orm(current_user)


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_update: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """更新當前用戶資訊"""
    try:
        user_service = UserService(db)
        updated_user = user_service.update_user(current_user.user_id, user_update)

        logger.info(f"用戶資訊更新: {current_user.email}")

        return UserResponse.from_orm(updated_user)

    except Exception as e:
        logger.error(f"用戶資訊更新失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="更新失敗，請稍後再試"
        )


@router.post("/change-password", response_model=Dict[str, Any])
async def change_password(
    password_change: PasswordChange,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """變更密碼"""
    try:
        # 驗證目前密碼
        if not current_user.verify_password(password_change.current_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="目前密碼錯誤"
            )

        # 更新密碼
        current_user.set_password(password_change.new_password)

        # 停用其他會話（保留當前會話）
        db.query(UserSession).filter(
            UserSession.user_id == current_user.user_id,
            UserSession.is_active == True
        ).update({"is_active": False, "logout_at": datetime.utcnow()})

        db.commit()

        logger.info(f"密碼變更成功: {current_user.email}")

        return {
            "success": True,
            "message": "密碼變更成功"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"密碼變更失敗: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="密碼變更失敗，請稍後再試"
        )