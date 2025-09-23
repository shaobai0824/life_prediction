"""
速率限制中間件

防止 API 濫用和 DDoS 攻擊
"""

from fastapi import Request, Response, HTTPException, status
from fastapi.responses import JSONResponse
import time
from typing import Dict, List
from app.core.config import settings


class RateLimitMiddleware:
    """速率限制中間件"""

    def __init__(self, app):
        self.app = app
        self._requests: Dict[str, List[float]] = {}

    async def __call__(self, request: Request, call_next):
        # 獲取客戶端識別（IP 地址）
        client_ip = self._get_client_ip(request)

        # 檢查速率限制
        if not self._is_allowed(client_ip):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "請求過於頻繁",
                    "message": f"每小時最多允許 {settings.RATE_LIMIT_REQUESTS} 個請求",
                    "retry_after": 3600
                },
                headers={"Retry-After": "3600"}
            )

        # 繼續處理請求
        response = await call_next(request)

        # 添加速率限制資訊到回應標頭
        remaining = self._get_remaining_requests(client_ip)
        response.headers["X-RateLimit-Limit"] = str(settings.RATE_LIMIT_REQUESTS)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + settings.RATE_LIMIT_WINDOW)

        return response

    def _get_client_ip(self, request: Request) -> str:
        """獲取客戶端 IP 地址"""
        # 檢查代理標頭
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # 使用客戶端 IP
        return request.client.host if request.client else "unknown"

    def _is_allowed(self, client_ip: str) -> bool:
        """檢查是否允許請求"""
        now = time.time()
        window_start = now - settings.RATE_LIMIT_WINDOW

        # 初始化或清理過期請求
        if client_ip not in self._requests:
            self._requests[client_ip] = []
        else:
            # 移除超出時間窗口的請求
            self._requests[client_ip] = [
                req_time for req_time in self._requests[client_ip]
                if req_time > window_start
            ]

        # 檢查是否超過限制
        if len(self._requests[client_ip]) >= settings.RATE_LIMIT_REQUESTS:
            return False

        # 記錄當前請求
        self._requests[client_ip].append(now)
        return True

    def _get_remaining_requests(self, client_ip: str) -> int:
        """獲取剩餘請求數"""
        if client_ip not in self._requests:
            return settings.RATE_LIMIT_REQUESTS

        current_requests = len(self._requests[client_ip])
        return max(0, settings.RATE_LIMIT_REQUESTS - current_requests)