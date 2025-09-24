"""
電子郵件服務

處理系統郵件發送功能
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Optional
import logging
import asyncio
from pathlib import Path

from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class EmailService:
    """電子郵件服務類"""

    def __init__(self):
        self.smtp_server = settings.SMTP_HOST
        self.smtp_port = settings.SMTP_PORT
        self.smtp_username = settings.SMTP_USERNAME
        self.smtp_password = settings.SMTP_PASSWORD
        self.from_email = settings.FROM_EMAIL
        self.from_name = settings.FROM_NAME

    def _create_smtp_connection(self):
        """建立SMTP連接"""
        try:
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            return server
        except Exception as e:
            logger.error(f"SMTP連接失敗: {e}")
            raise

    def send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: Optional[str] = None,
        attachments: Optional[List[str]] = None
    ) -> bool:
        """發送電子郵件"""
        try:
            # 建立郵件對象
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = f"{self.from_name} <{self.from_email}>"
            msg['To'] = to_email

            # 文字內容
            if text_content:
                text_part = MIMEText(text_content, 'plain', 'utf-8')
                msg.attach(text_part)

            # HTML內容
            html_part = MIMEText(html_content, 'html', 'utf-8')
            msg.attach(html_part)

            # 附件
            if attachments:
                for file_path in attachments:
                    self._add_attachment(msg, file_path)

            # 發送郵件
            with self._create_smtp_connection() as server:
                server.send_message(msg)

            logger.info(f"郵件發送成功: {to_email}")
            return True

        except Exception as e:
            logger.error(f"郵件發送失敗: {e}")
            return False

    def _add_attachment(self, msg: MIMEMultipart, file_path: str):
        """添加附件"""
        try:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"附件文件不存在: {file_path}")
                return

            with open(file_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())

            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename= {path.name}',
            )
            msg.attach(part)

        except Exception as e:
            logger.error(f"添加附件失敗: {e}")

    def _load_template(self, template_name: str, **kwargs) -> str:
        """載入郵件模板"""
        try:
            template_path = Path(__file__).parent.parent / "templates" / "email" / f"{template_name}.html"

            if not template_path.exists():
                logger.warning(f"郵件模板不存在: {template_name}")
                return self._get_default_template(**kwargs)

            with open(template_path, 'r', encoding='utf-8') as f:
                template = f.read()

            # 簡單的模板替換
            for key, value in kwargs.items():
                template = template.replace(f"{{{{{key}}}}}", str(value))

            return template

        except Exception as e:
            logger.error(f"載入郵件模板失敗: {e}")
            return self._get_default_template(**kwargs)

    def _get_default_template(self, **kwargs) -> str:
        """預設郵件模板"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>壽命預測系統</title>
        </head>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: linear-gradient(135deg, #d4af37 0%, #ffd700 100%); padding: 20px; text-align: center;">
                <h1 style="color: #8b4513; margin: 0;">🔮 壽命預測系統</h1>
                <p style="color: #8b4513; margin: 10px 0;">專業的面相分析與壽命預測服務</p>
            </div>
            <div style="padding: 30px; background: #f9f9f9;">
                {kwargs.get('content', '郵件內容')}
            </div>
            <div style="background: #333; color: #fff; text-align: center; padding: 20px; font-size: 12px;">
                <p>© 2024 壽命預測系統 - 僅供參考，不構成醫療建議</p>
                <p>如有疑問，請聯繫客服：support@life-prediction.com</p>
            </div>
        </body>
        </html>
        """


# 全域郵件服務實例
email_service = EmailService()


async def send_verification_email(email: str, username: str, verification_code: str, token: str) -> bool:
    """發送驗證郵件"""
    subject = "🔮 歡迎加入壽命預測系統 - 請驗證您的電子郵件"

    content = f"""
    <h2 style="color: #d4af37;">歡迎，{username}！</h2>
    <p>感謝您註冊壽命預測系統。請使用以下驗證碼完成電子郵件驗證：</p>

    <div style="background: #fff; border: 2px solid #d4af37; border-radius: 10px; padding: 20px; text-align: center; margin: 20px 0;">
        <h3 style="color: #8b4513; font-size: 32px; margin: 0; letter-spacing: 5px;">{verification_code}</h3>
        <p style="color: #666; margin: 10px 0;">此驗證碼將在24小時後過期</p>
    </div>

    <p><strong>注意事項：</strong></p>
    <ul style="color: #666;">
        <li>此驗證碼僅供一次性使用</li>
        <li>請勿與他人分享您的驗證碼</li>
        <li>如非本人操作，請忽略此郵件</li>
    </ul>

    <div style="margin-top: 30px; padding: 15px; background: #fff3cd; border-left: 4px solid #d4af37;">
        <p style="margin: 0; color: #856404;">
            <strong>溫馨提醒：</strong>我們的預測結果僅供參考，不構成專業醫療建議。
            如有健康疑慮，請諮詢專業醫療人員。
        </p>
    </div>
    """

    html_content = email_service._get_default_template(content=content)

    return email_service.send_email(
        to_email=email,
        subject=subject,
        html_content=html_content
    )


async def send_password_reset_email(email: str, username: str, verification_code: str, token: str) -> bool:
    """發送密碼重設郵件"""
    subject = "🔮 壽命預測系統 - 密碼重設驗證"

    content = f"""
    <h2 style="color: #d4af37;">密碼重設請求</h2>
    <p>親愛的 {username}，</p>
    <p>我們收到了您的密碼重設請求。請使用以下驗證碼完成密碼重設：</p>

    <div style="background: #fff; border: 2px solid #d4af37; border-radius: 10px; padding: 20px; text-align: center; margin: 20px 0;">
        <h3 style="color: #8b4513; font-size: 32px; margin: 0; letter-spacing: 5px;">{verification_code}</h3>
        <p style="color: #666; margin: 10px 0;">此驗證碼將在1小時後過期</p>
    </div>

    <p><strong>安全提醒：</strong></p>
    <ul style="color: #666;">
        <li>如果您沒有請求密碼重設，請忽略此郵件</li>
        <li>請勿與他人分享您的驗證碼</li>
        <li>建議設定強密碼以保護帳戶安全</li>
    </ul>

    <div style="margin-top: 30px; padding: 15px; background: #f8d7da; border-left: 4px solid #dc3545;">
        <p style="margin: 0; color: #721c24;">
            <strong>安全警告：</strong>如果您經常收到此類郵件，可能表示有人正在嘗試入侵您的帳戶。
            請立即聯繫我們的客服團隊。
        </p>
    </div>
    """

    html_content = email_service._get_default_template(content=content)

    return email_service.send_email(
        to_email=email,
        subject=subject,
        html_content=html_content
    )


async def send_welcome_email(email: str, username: str) -> bool:
    """發送歡迎郵件"""
    subject = "🔮 歡迎來到壽命預測系統！"

    content = f"""
    <h2 style="color: #d4af37;">歡迎，{username}！</h2>
    <p>恭喜您成功加入壽命預測系統！現在您可以開始體驗我們專業的面相分析服務。</p>

    <div style="background: #fff; border: 2px solid #d4af37; border-radius: 10px; padding: 20px; margin: 20px 0;">
        <h3 style="color: #8b4513;">🎁 新用戶專享</h3>
        <p>作為新用戶，您將獲得：</p>
        <ul style="color: #333;">
            <li>✨ <strong>3次免費預測</strong>機會</li>
            <li>📊 <strong>詳細分析報告</strong></li>
            <li>🎯 <strong>個人化建議</strong></li>
            <li>🔒 <strong>隱私保護</strong>承諾</li>
        </ul>
    </div>

    <div style="text-align: center; margin: 30px 0;">
        <a href="#" style="background: linear-gradient(135deg, #d4af37, #ffd700); color: white; padding: 15px 30px; text-decoration: none; border-radius: 25px; font-weight: bold; display: inline-block;">
            🚀 開始第一次預測
        </a>
    </div>

    <div style="margin-top: 30px; padding: 15px; background: #d1ecf1; border-left: 4px solid #17a2b8;">
        <p style="margin: 0; color: #0c5460;">
            <strong>使用指南：</strong><br>
            1. 上傳清晰的正面照片<br>
            2. 等待AI分析處理<br>
            3. 獲得專業預測報告<br>
            4. 參考建議優化生活
        </p>
    </div>

    <div style="margin-top: 30px; padding: 15px; background: #fff3cd; border-left: 4px solid #d4af37;">
        <p style="margin: 0; color: #856404;">
            <strong>重要聲明：</strong>本系統提供的預測結果僅供娛樂和參考，不構成專業醫療建議。
            如有健康疑慮，請諮詢合格的醫療專業人員。
        </p>
    </div>
    """

    html_content = email_service._get_default_template(content=content)

    return email_service.send_email(
        to_email=email,
        subject=subject,
        html_content=html_content
    )


async def send_prediction_report_email(email: str, username: str, report_data: dict) -> bool:
    """發送預測報告郵件"""
    subject = f"🔮 您的壽命預測報告已生成 - {username}"

    predicted_age = report_data.get('predicted_lifespan', '未知')
    confidence = report_data.get('confidence_level', '未知')

    content = f"""
    <h2 style="color: #d4af37;">您的預測報告</h2>
    <p>親愛的 {username}，</p>
    <p>您的面相分析已完成，以下是您的預測報告：</p>

    <div style="background: #fff; border: 2px solid #d4af37; border-radius: 10px; padding: 20px; margin: 20px 0;">
        <h3 style="color: #8b4513; text-align: center;">預測結果</h3>
        <div style="text-align: center;">
            <p style="font-size: 24px; color: #d4af37; margin: 10px 0;"><strong>{predicted_age} 歲</strong></p>
            <p style="color: #666;">信心度：{confidence}</p>
        </div>
    </div>

    <div style="margin: 20px 0;">
        <h4 style="color: #8b4513;">健康建議</h4>
        <ul style="color: #333;">
            <li>保持規律運動習慣</li>
            <li>維持均衡營養飲食</li>
            <li>確保充足優質睡眠</li>
            <li>定期健康檢查</li>
        </ul>
    </div>

    <div style="text-align: center; margin: 30px 0;">
        <a href="#" style="background: linear-gradient(135deg, #d4af37, #ffd700); color: white; padding: 12px 24px; text-decoration: none; border-radius: 20px; font-weight: bold; display: inline-block;">
            📊 查看完整報告
        </a>
    </div>

    <div style="margin-top: 30px; padding: 15px; background: #fff3cd; border-left: 4px solid #d4af37;">
        <p style="margin: 0; color: #856404;">
            <strong>免責聲明：</strong>此預測結果僅基於面相分析算法，不構成醫療診斷或建議。
            實際壽命受多種因素影響，包括但不限於遺傳、生活方式、環境和醫療護理。
        </p>
    </div>
    """

    html_content = email_service._get_default_template(content=content)

    return email_service.send_email(
        to_email=email,
        subject=subject,
        html_content=html_content
    )