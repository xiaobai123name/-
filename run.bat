@echo off
echo ============================================
echo    智能学习伴侣系统 - 启动中...
echo ============================================
echo.

REM 激活虚拟环境
call venv\Scripts\activate.bat

REM 检查.env文件
if not exist .env (
    echo [警告] 未找到 .env 文件，正在从模板创建...
    copy .env.example .env
    echo [提示] 请编辑 .env 文件，填入您的 API 密钥
    echo.
)

REM 启动应用
echo 正在启动 Streamlit 应用...
echo 访问地址: http://localhost:8501
echo.
streamlit run app/主页.py

pause
