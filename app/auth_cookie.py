"""
Streamlit 登录持久化工具（Cookie + URL token 双保险）。
"""

from datetime import datetime, timedelta
import hashlib
import hmac
import secrets
import time
from typing import Literal, Optional, Tuple

import streamlit as st

from backend.config import settings
from backend.database.crud import DatabaseManager

COOKIE_KEY = "app_user_id"
COOKIE_WIDGET_KEY = "app_cookie_manager"
COOKIE_EXPIRE_DAYS = 7

QUERY_AUTH_KEY = "auth_token"
TOKEN_TTL_SECONDS = COOKIE_EXPIRE_DAYS * 24 * 60 * 60
AUTH_SECRET_PATH = settings.database_path.parent / ".auth_signing_key"

_SIGNING_SECRET: Optional[bytes] = None

try:
    import extra_streamlit_components as stx

    COOKIE_MANAGER_AVAILABLE = True
except ImportError:
    COOKIE_MANAGER_AVAILABLE = False
    stx = None


AuthRestoreStatus = Literal["ready", "restored", "pending", "missing", "disabled", "unavailable"]


def ensure_auth_state_defaults() -> None:
    """确保认证相关会话键存在。"""
    if "user" not in st.session_state:
        st.session_state.user = None
    if "cookie_login_disabled" not in st.session_state:
        st.session_state.cookie_login_disabled = False
    if "_auth_cookie_probe_round" not in st.session_state:
        st.session_state._auth_cookie_probe_round = 0


def _get_cookie_manager():
    """获取 CookieManager（在 session_state 中缓存）。"""
    if not COOKIE_MANAGER_AVAILABLE:
        return None
    if "_auth_cookie_manager" not in st.session_state:
        st.session_state._auth_cookie_manager = stx.CookieManager(key=COOKIE_WIDGET_KEY)
    return st.session_state._auth_cookie_manager


def _load_signing_secret() -> bytes:
    """加载（或生成）本地签名密钥。"""
    global _SIGNING_SECRET
    if _SIGNING_SECRET is not None:
        return _SIGNING_SECRET

    AUTH_SECRET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if AUTH_SECRET_PATH.exists():
        raw = AUTH_SECRET_PATH.read_text(encoding="utf-8").strip()
        if raw:
            _SIGNING_SECRET = raw.encode("utf-8")
            return _SIGNING_SECRET

    generated = secrets.token_hex(32)
    AUTH_SECRET_PATH.write_text(generated, encoding="utf-8")
    _SIGNING_SECRET = generated.encode("utf-8")
    return _SIGNING_SECRET


def _build_auth_token(user_id: str, ts: Optional[int] = None) -> str:
    timestamp = int(ts if ts is not None else time.time())
    message = f"{user_id}.{timestamp}"
    signature = hmac.new(_load_signing_secret(), message.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{user_id}.{timestamp}.{signature}"


def _decode_auth_token(token: str) -> Optional[Tuple[str, int]]:
    if not token:
        return None
    if "." not in token:
        return None

    try:
        user_id, ts_text, signature = token.rsplit(".", 2)
        timestamp = int(ts_text)
    except Exception:
        return None

    # 过期校验
    now = int(time.time())
    if timestamp <= 0 or (now - timestamp) > TOKEN_TTL_SECONDS:
        return None

    message = f"{user_id}.{timestamp}"
    expected = hmac.new(_load_signing_secret(), message.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None

    return user_id, timestamp


def _read_query_auth_token() -> str:
    raw = st.query_params.get(QUERY_AUTH_KEY, "")
    if isinstance(raw, list):
        return str(raw[0]) if raw else ""
    return str(raw or "")


def _set_query_auth_token_for_user(user_id: str) -> None:
    """确保 URL 中存在当前用户的有效 token。"""
    existing = _read_query_auth_token()
    decoded = _decode_auth_token(existing) if existing else None
    if decoded and decoded[0] == str(user_id):
        return
    st.query_params[QUERY_AUTH_KEY] = _build_auth_token(str(user_id))


def _clear_query_auth_token() -> None:
    try:
        if QUERY_AUTH_KEY in st.query_params:
            del st.query_params[QUERY_AUTH_KEY]
    except Exception:
        pass


def _set_user_session(user) -> None:
    st.session_state.user = {
        "id": user.id,
        "username": user.username,
        "display_name": user.display_name,
    }
    st.session_state._auth_cookie_probe_round = 0
    _set_query_auth_token_for_user(user.id)


def _try_restore_from_query_token(db_manager: DatabaseManager) -> bool:
    token = _read_query_auth_token()
    decoded = _decode_auth_token(token) if token else None
    if not decoded:
        return False

    user_id, _ = decoded
    user = db_manager.get_user_by_id(user_id)
    if not user:
        return False

    _set_user_session(user)
    return True


def restore_user_from_cookie(db_manager: DatabaseManager) -> AuthRestoreStatus:
    """
    尝试恢复登录态（优先 cookie，失败则回退 URL token）。

    返回:
        ready: session_state 已有 user
        restored: 本次成功恢复
        pending: 首次探测 cookie，建议当前页 st.stop() 等待组件自动 rerun
        missing: 未找到可恢复登录态
        disabled: 用户主动登出后，禁止自动恢复
        unavailable: CookieManager 不可用（且 URL token 也不可用）
    """
    ensure_auth_state_defaults()

    if st.session_state.user is not None:
        _set_query_auth_token_for_user(st.session_state.user["id"])
        return "ready"

    if st.session_state.get("cookie_login_disabled", False):
        return "disabled"

    cookie_manager = _get_cookie_manager()
    if cookie_manager:
        user_id = cookie_manager.get(COOKIE_KEY)
        if user_id:
            user = db_manager.get_user_by_id(user_id)
            if user:
                _set_user_session(user)
                return "restored"

    # 回退：用 URL 签名 token 恢复（应对第三方 cookie 在刷新场景不稳定）
    if _try_restore_from_query_token(db_manager):
        return "restored"

    # cookie 首轮空值时给组件一次自动 rerun 机会
    if cookie_manager:
        probe_round = int(st.session_state.get("_auth_cookie_probe_round", 0))
        if probe_round < 1:
            st.session_state._auth_cookie_probe_round = probe_round + 1
            return "pending"

    return "missing" if cookie_manager else "unavailable"


def save_login_cookie(user_id: str) -> None:
    """写入登录持久态（URL token + cookie）。"""
    st.session_state.cookie_login_disabled = False
    st.session_state._auth_cookie_probe_round = 0
    _set_query_auth_token_for_user(str(user_id))

    cookie_manager = _get_cookie_manager()
    if not cookie_manager:
        return

    cookie_manager.set(
        COOKIE_KEY,
        str(user_id),
        expires_at=datetime.now() + timedelta(days=COOKIE_EXPIRE_DAYS),
    )


def clear_login_cookie() -> None:
    """清理登录持久态（URL token + cookie）。"""
    st.session_state._auth_cookie_probe_round = 0
    _clear_query_auth_token()

    cookie_manager = _get_cookie_manager()
    if not cookie_manager:
        return

    try:
        cookie_manager.delete(COOKIE_KEY)
    except Exception:
        pass

    try:
        cookie_manager.set(COOKIE_KEY, "", expires_at=datetime.now() - timedelta(days=1))
    except Exception:
        pass
