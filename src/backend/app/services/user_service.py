"""
用戶服務層

處理用戶相關的業務邏輯
"""

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from typing import Optional, List
from datetime import datetime, timedelta
import logging

from ..models.user import User, UserPreferences
from ..schemas.user import UserCreate, UserUpdate
from ..core.security import get_password_hash

logger = logging.getLogger(__name__)


class UserService:
    """用戶服務類"""

    def __init__(self, db: Session):
        self.db = db

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """根據用戶ID獲取用戶"""
        return self.db.query(User).filter(User.user_id == user_id).first()

    def get_user_by_email(self, email: str) -> Optional[User]:
        """根據電子郵件獲取用戶"""
        return self.db.query(User).filter(User.email == email).first()

    def get_user_by_username(self, username: str) -> Optional[User]:
        """根據用戶名獲取用戶"""
        return self.db.query(User).filter(User.username == username).first()

    def create_user(self, user_create: UserCreate) -> User:
        """建立新用戶"""
        try:
            # 建立用戶物件
            db_user = User(
                email=user_create.email,
                username=user_create.username,
                full_name=user_create.full_name,
                phone=user_create.phone,
                preferred_language=user_create.preferred_language or "zh-TW",
                consent_data_processing=user_create.consent_data_processing,
                consent_marketing=user_create.consent_marketing or False,
                consent_analytics=user_create.consent_analytics or False
            )

            # 設定密碼
            db_user.set_password(user_create.password)

            # 新用戶預設狀態
            db_user.is_active = True
            db_user.is_verified = False  # 需要電子郵件驗證
            db_user.remaining_predictions = 3  # 免費用戶3次預測

            self.db.add(db_user)
            self.db.commit()
            self.db.refresh(db_user)

            logger.info(f"用戶建立成功: {db_user.email}")
            return db_user

        except IntegrityError as e:
            self.db.rollback()
            logger.error(f"建立用戶失敗 - 資料庫完整性錯誤: {e}")
            raise ValueError("電子郵件或用戶名已被使用")
        except Exception as e:
            self.db.rollback()
            logger.error(f"建立用戶失敗: {e}")
            raise

    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """驗證用戶登入"""
        try:
            user = self.get_user_by_email(email)
            if not user:
                logger.warning(f"登入嘗試失敗 - 用戶不存在: {email}")
                return None

            if not user.verify_password(password):
                logger.warning(f"登入嘗試失敗 - 密碼錯誤: {email}")
                return None

            logger.info(f"用戶認證成功: {email}")
            return user

        except Exception as e:
            logger.error(f"用戶認證失敗: {e}")
            return None

    def update_user(self, user_id: str, user_update: UserUpdate) -> User:
        """更新用戶資訊"""
        try:
            user = self.get_user_by_id(user_id)
            if not user:
                raise ValueError("用戶不存在")

            # 更新允許的欄位
            update_data = user_update.dict(exclude_unset=True)
            for field, value in update_data.items():
                if hasattr(user, field):
                    setattr(user, field, value)

            user.updated_at = datetime.utcnow()

            self.db.commit()
            self.db.refresh(user)

            logger.info(f"用戶資訊更新成功: {user.email}")
            return user

        except Exception as e:
            self.db.rollback()
            logger.error(f"用戶資訊更新失敗: {e}")
            raise

    def deactivate_user(self, user_id: str) -> bool:
        """停用用戶"""
        try:
            user = self.get_user_by_id(user_id)
            if not user:
                return False

            user.is_active = False
            user.updated_at = datetime.utcnow()

            self.db.commit()

            logger.info(f"用戶已停用: {user.email}")
            return True

        except Exception as e:
            self.db.rollback()
            logger.error(f"停用用戶失敗: {e}")
            return False

    def activate_user(self, user_id: str) -> bool:
        """啟用用戶"""
        try:
            user = self.get_user_by_id(user_id)
            if not user:
                return False

            user.is_active = True
            user.updated_at = datetime.utcnow()

            self.db.commit()

            logger.info(f"用戶已啟用: {user.email}")
            return True

        except Exception as e:
            self.db.rollback()
            logger.error(f"啟用用戶失敗: {e}")
            return False

    def verify_user_email(self, user_id: str) -> bool:
        """驗證用戶電子郵件"""
        try:
            user = self.get_user_by_id(user_id)
            if not user:
                return False

            user.is_verified = True
            user.updated_at = datetime.utcnow()

            self.db.commit()

            logger.info(f"用戶電子郵件已驗證: {user.email}")
            return True

        except Exception as e:
            self.db.rollback()
            logger.error(f"電子郵件驗證失敗: {e}")
            return False

    def upgrade_to_premium(self, user_id: str) -> bool:
        """升級為高級用戶"""
        try:
            user = self.get_user_by_id(user_id)
            if not user:
                return False

            user.is_premium = True
            user.remaining_predictions = -1  # 無限制
            user.updated_at = datetime.utcnow()

            self.db.commit()

            logger.info(f"用戶已升級為高級會員: {user.email}")
            return True

        except Exception as e:
            self.db.rollback()
            logger.error(f"升級高級會員失敗: {e}")
            return False

    def use_prediction_quota(self, user_id: str) -> bool:
        """使用預測額度"""
        try:
            user = self.get_user_by_id(user_id)
            if not user:
                return False

            if not user.can_predict():
                return False

            user.use_prediction()
            user.updated_at = datetime.utcnow()

            self.db.commit()

            logger.info(f"用戶使用預測額度: {user.email}, 剩餘: {user.remaining_predictions}")
            return True

        except Exception as e:
            self.db.rollback()
            logger.error(f"使用預測額度失敗: {e}")
            return False

    def get_user_statistics(self, user_id: str) -> dict:
        """獲取用戶統計資訊"""
        try:
            user = self.get_user_by_id(user_id)
            if not user:
                return {}

            # 計算本月預測次數
            month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            this_month_predictions = user.total_predictions  # 簡化版本

            return {
                "total_predictions": user.total_predictions,
                "remaining_predictions": user.remaining_predictions if not user.is_premium else -1,
                "this_month_predictions": this_month_predictions,
                "member_since": user.created_at,
                "last_prediction": None,  # 需要從預測記錄表獲取
                "accuracy_score": None,  # 需要從預測結果計算
                "is_premium": user.is_premium,
                "is_verified": user.is_verified
            }

        except Exception as e:
            logger.error(f"獲取用戶統計失敗: {e}")
            return {}

    def get_user_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """獲取用戶偏好設定"""
        return self.db.query(UserPreferences).filter(
            UserPreferences.user_id == user_id
        ).first()

    def update_user_preferences(self, user_id: str, preferences_data: dict) -> UserPreferences:
        """更新用戶偏好設定"""
        try:
            preferences = self.get_user_preferences(user_id)

            if not preferences:
                # 建立新的偏好設定
                preferences = UserPreferences(user_id=user_id)
                self.db.add(preferences)

            # 更新偏好設定
            for field, value in preferences_data.items():
                if hasattr(preferences, field) and value is not None:
                    setattr(preferences, field, value)

            preferences.updated_at = datetime.utcnow()

            self.db.commit()
            self.db.refresh(preferences)

            logger.info(f"用戶偏好設定更新成功: {user_id}")
            return preferences

        except Exception as e:
            self.db.rollback()
            logger.error(f"用戶偏好設定更新失敗: {e}")
            raise

    def search_users(self, query: str, limit: int = 20) -> List[User]:
        """搜尋用戶（管理功能）"""
        try:
            return self.db.query(User).filter(
                User.email.contains(query) |
                User.username.contains(query) |
                User.full_name.contains(query)
            ).limit(limit).all()

        except Exception as e:
            logger.error(f"搜尋用戶失敗: {e}")
            return []

    def get_recent_users(self, days: int = 7, limit: int = 50) -> List[User]:
        """獲取最近註冊的用戶"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            return self.db.query(User).filter(
                User.created_at >= cutoff_date
            ).order_by(User.created_at.desc()).limit(limit).all()

        except Exception as e:
            logger.error(f"獲取最近用戶失敗: {e}")
            return []