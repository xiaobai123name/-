# GitHub 上传与更新教程（Windows）

适用场景：把本地项目（例如 `D:\my_share\bishe`）上传到 GitHub 仓库，并在后续持续更新。

## 0. 准备

- 安装 Git（终端能运行 `git --version`）。
- 确认你的 GitHub 账号已登录网页端，并且你知道要推到哪个仓库（例如：`https://github.com/<用户名>/<仓库名>.git`）。

## 1. 在 GitHub 网页创建新仓库

1. 打开 GitHub 主页（右上角 `+` 或左侧 `New`）。
2. 填写 `Repository name`，选择 `Public/Private`。
3. 建议：如果你本地已经有完整项目，先不要勾选 `Add a README file`（避免和本地初始化产生冲突）。
4. 点击 `Create repository`。
5. 进入仓库页面后，点 `Code` 复制 **HTTPS** 仓库地址，形如：`https://github.com/<用户名>/<仓库名>.git`。

## 2. 第一次上传（把本地项目推到远程）

在项目根目录执行（每行是一条命令，不要粘成一行）：

```bash
cd D:\my_share\bishe
git init
git add .
git commit -m "init"
git branch -M main
git remote add origin https://github.com/<用户名>/<仓库名>.git
git push -u origin main
```

## 3. 后续更新（改完代码再推送）

```bash
cd D:\my_share\bishe
git status
git add .
git commit -m "update"
git push
```

## 4. 常见报错与处理

### 4.1 `Failed to connect to 127.0.0.1 port 7890: Connection refused`

原因：Git 被配置走本地代理 `127.0.0.1:7890`，但代理软件没开/端口不对。

处理（不使用代理时，取消 Git 全局代理）：

```bash
git config --global --unset http.proxy
git config --global --unset https.proxy
git config --global --unset all.proxy
```

### 4.2 `remote: Repository not found`

原因通常是：
- 远程仓库地址写错（建议从仓库页面 `Code` 复制 HTTPS 地址）。
- 仓库不存在（没创建成功）。
- 私有仓库但你未登录/无权限。

处理：
- 检查当前远程地址：`git remote -v`
- 修正远程地址：`git remote set-url origin https://github.com/<用户名>/<仓库名>.git`

### 4.3 `Permission denied ... 403`

原因：你当前使用的 GitHub 账号（或缓存的凭证）没有该仓库权限；常见于“仓库属于 A 账号，但你用 B 账号在推”。

处理：
- 确认远程仓库属于你当前账号，或已把你当前账号加为协作者（collaborator）。
- 或清理凭证后重新推送。

### 4.4 私有仓库推送要 Token（GitHub 不再支持密码）

推送时如果提示输入 Username/Password：
- Username：你的 GitHub 用户名
- Password：不是密码，而是 **Personal Access Token (PAT)**

获取 Token（网页端）：
1. 打开 `https://github.com/settings/tokens`
2. 选择 `Generate new token (classic)`（最省事）
3. 勾选权限：`repo`
4. 生成后复制 Token（只显示一次）

安全提醒：
- Token 等同密码，不要发给任何人，也不要提交进仓库（例如 `.env`、配置文件里）。
- 如果不小心泄露，立刻在 GitHub 里删除/撤销该 Token，然后重新生成。

