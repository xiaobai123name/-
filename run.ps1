# 智能学习伴侣系统 - PowerShell 启动脚本

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   智能学习伴侣系统 - 启动中..." -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# 激活虚拟环境
& .\venv\Scripts\Activate.ps1

# 检查.env文件
if (-not (Test-Path ".env")) {
    Write-Host "[警告] 未找到 .env 文件，正在从模板创建..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host "[提示] 请编辑 .env 文件，填入您的 API 密钥" -ForegroundColor Yellow
    Write-Host ""
}

# 启动应用
Write-Host "正在启动 Streamlit 应用..." -ForegroundColor Green
Write-Host "访问地址: http://localhost:8501" -ForegroundColor Green
Write-Host ""

streamlit run app/主页.py
