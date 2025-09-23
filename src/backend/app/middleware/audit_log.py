"""
審計日誌中間件

符合金融業合規要求的完整操作記錄
"""

import json
import time
import uuid
from datetime import datetime
from fastapi import Request, Response
from app.core.config import settings
import logging

# 配置審計日誌記錄器
audit_logger = logging.getLogger("audit")
audit_handler = logging.FileHandler("./logs/audit.log")
audit_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
audit_handler.setFormatter(audit_formatter)
audit_logger.addHandler(audit_handler)
audit_logger.setLevel(logging.INFO)


class AuditLogMiddleware:
    """審計日誌中間件"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, request: Request, call_next):
        # 生成請求 ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # 記錄請求開始
        start_time = time.time()

        # 收集請求資訊
        request_info = await self._collect_request_info(request, request_id)

        # 記錄請求
        if self._should_audit_request(request):
            audit_logger.info(f"REQUEST: {json.dumps(request_info, ensure_ascii=False)}")

        try:
            # 處理請求
            response = await call_next(request)

            # 計算處理時間
            process_time = time.time() - start_time

            # 收集回應資訊
            response_info = self._collect_response_info(
                response, request_id, process_time, "SUCCESS"
            )

            # 記錄回應
            if self._should_audit_request(request):
                audit_logger.info(f"RESPONSE: {json.dumps(response_info, ensure_ascii=False)}")

            return response

        except Exception as e:
            # 計算處理時間
            process_time = time.time() - start_time

            # 記錄錯誤
            error_info = self._collect_response_info(
                None, request_id, process_time, "ERROR", str(e)
            )

            audit_logger.error(f"ERROR: {json.dumps(error_info, ensure_ascii=False)}")

            # 重新拋出例外
            raise

    async def _collect_request_info(self, request: Request, request_id: str) -> dict:
        """收集請求資訊"""
        # 獲取客戶端 IP
        client_ip = self._get_client_ip(request)

        # 基本請求資訊
        info = {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": client_ip,
            "user_agent": request.headers.get("User-Agent", ""),
            "headers": self._filter_sensitive_headers(dict(request.headers))
        }

        # 添加用戶資訊（如果已認證）
        if hasattr(request.state, "user"):
            info["user_id"] = getattr(request.state.user, "id", None)
            info["username"] = getattr(request.state.user, "username", None)

        # 對於特定端點，記錄請求體（排除敏感資訊）
        if self._should_log_body(request):
            try:
                body = await request.body()
                if body:
                    # 如果是 JSON，解析並過濾敏感欄位
                    if request.headers.get("content-type", "").startswith("application/json"):
                        body_json = json.loads(body.decode())
                        info["body"] = self._filter_sensitive_data(body_json)
                    else:
                        info["body_size"] = len(body)
            except Exception:
                info["body"] = "無法解析請求體"

        return info

    def _collect_response_info(
        self,
        response: Response,
        request_id: str,
        process_time: float,
        status: str,
        error_message: str = None
    ) -> dict:
        """收集回應資訊"""
        info = {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status": status,
            "process_time": round(process_time, 4)
        }

        if response:
            info["status_code"] = response.status_code
            info["headers"] = self._filter_sensitive_headers(dict(response.headers))

        if error_message:
            info["error"] = error_message

        return info

    def _get_client_ip(self, request: Request) -> str:
        """獲取客戶端 IP 地址"""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        return request.client.host if request.client else "unknown"

    def _should_audit_request(self, request: Request) -> bool:
        """判斷是否需要審計此請求"""
        if not settings.AUDIT_LOG_ENABLED:
            return False

        # 排除靜態檔案和健康檢查
        excluded_paths = ["/docs", "/redoc", "/openapi.json", "/favicon.ico", "/static"]

        # 健康檢查端點不需要詳細審計
        if request.url.path.startswith("/health"):
            return False

        # 檢查是否在排除列表中
        for excluded in excluded_paths:
            if request.url.path.startswith(excluded):
                return False

        return True

    def _should_log_body(self, request: Request) -> bool:
        """判斷是否需要記錄請求體"""
        # 只對特定端點記錄請求體
        sensitive_endpoints = ["/api/v1/auth", "/api/v1/predict"]

        for endpoint in sensitive_endpoints:
            if request.url.path.startswith(endpoint):
                return True

        return False

    def _filter_sensitive_headers(self, headers: dict) -> dict:
        """過濾敏感標頭"""
        sensitive_headers = {
            "authorization", "cookie", "x-api-key", "x-auth-token"
        }

        filtered = {}
        for key, value in headers.items():
            if key.lower() in sensitive_headers:
                filtered[key] = "[REDACTED]"
            else:
                filtered[key] = value

        return filtered

    def _filter_sensitive_data(self, data: dict) -> dict:
        """過濾敏感資料"""
        sensitive_fields = {
            "password", "token", "secret", "key", "credential"
        }

        filtered = {}
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_fields):
                filtered[key] = "[REDACTED]"
            elif isinstance(value, dict):
                filtered[key] = self._filter_sensitive_data(value)
            else:
                filtered[key] = value

        return filtered