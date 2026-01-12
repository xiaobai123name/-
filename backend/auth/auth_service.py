"""
用户认证服务
提供用户注册、登录、密码验证等功能
"""

import bcrypt
from typing import Optional, Tuple
from datetime import datetime

from ..database.models import User
from ..database.crud import DatabaseManager


def get_password_hash(password: str) -> str:
    """
    生成密码哈希
    
    Args:
        password: 明文密码
        
    Returns:
        str: 密码哈希值
    """
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')


def verify_password(password: str, password_hash: str) -> bool:
    """
    验证密码
    
    Args:
        password: 明文密码
        password_hash: 存储的密码哈希
        
    Returns:
        bool: 密码是否正确
    """
    return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))


class AuthService:
    """认证服务类"""
    
    def __init__(self, db_manager: DatabaseManager):
        """
        初始化认证服务
        
        Args:
            db_manager: 数据库管理器实例
        """
        self.db = db_manager
    
    def register(
        self, 
        username: str, 
        password: str, 
        display_name: Optional[str] = None
    ) -> Tuple[bool, str, Optional[User]]:
        """
        用户注册
        
        Args:
            username: 用户名
            password: 密码
            display_name: 显示名称
            
        Returns:
            Tuple[bool, str, Optional[User]]: (是否成功, 消息, 用户对象)
        """
        # 验证用户名
        if not username or len(username) < 3:
            return False, "用户名至少需要3个字符", None
        
        if len(username) > 50:
            return False, "用户名不能超过50个字符", None
        
        # 验证密码
        if not password or len(password) < 6:
            return False, "密码至少需要6个字符", None
        
        # 检查用户名是否已存在
        existing_user = self.db.get_user_by_username(username)
        if existing_user:
            return False, "用户名已存在", None
        
        # 创建用户
        try:
            password_hash = get_password_hash(password)
            user = self.db.create_user(
                username=username,
                password_hash=password_hash,
                display_name=display_name
            )
            return True, "注册成功", user
        except Exception as e:
            return False, f"注册失败: {str(e)}", None
    
    def login(self, username: str, password: str) -> Tuple[bool, str, Optional[User]]:
        """
        用户登录
        
        Args:
            username: 用户名
            password: 密码
            
        Returns:
            Tuple[bool, str, Optional[User]]: (是否成功, 消息, 用户对象)
        """
        if not username or not password:
            return False, "请输入用户名和密码", None
        
        # 查找用户
        user = self.db.get_user_by_username(username)
        if not user:
            return False, "用户名或密码错误", None
        
        # 检查用户状态
        if not user.is_active:
            return False, "账户已被禁用", None
        
        # 验证密码
        if not verify_password(password, user.password_hash):
            return False, "用户名或密码错误", None
        
        # 更新登录时间
        self.db.update_user_login(user.id)
        
        return True, "登录成功", user
    
    def change_password(
        self, 
        user_id: str, 
        old_password: str, 
        new_password: str
    ) -> Tuple[bool, str]:
        """
        修改密码
        
        Args:
            user_id: 用户ID
            old_password: 旧密码
            new_password: 新密码
            
        Returns:
            Tuple[bool, str]: (是否成功, 消息)
        """
        user = self.db.get_user_by_id(user_id)
        if not user:
            return False, "用户不存在"
        
        # 验证旧密码
        if not verify_password(old_password, user.password_hash):
            return False, "旧密码错误"
        
        # 验证新密码
        if len(new_password) < 6:
            return False, "新密码至少需要6个字符"
        
        # 更新密码
        try:
            with self.db.get_session() as session:
                db_user = session.query(User).filter(User.id == user_id).first()
                if db_user:
                    db_user.password_hash = get_password_hash(new_password)
            return True, "密码修改成功"
        except Exception as e:
            return False, f"密码修改失败: {str(e)}"
    
    def get_user_info(self, user_id: str) -> Optional[dict]:
        """
        获取用户信息
        
        Args:
            user_id: 用户ID
            
        Returns:
            Optional[dict]: 用户信息字典
        """
        user = self.db.get_user_by_id(user_id)
        if not user:
            return None
        
        return {
            "id": user.id,
            "username": user.username,
            "display_name": user.display_name,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "last_login": user.last_login.isoformat() if user.last_login else None,
            "is_active": user.is_active
        }
