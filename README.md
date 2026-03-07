# 智能学习伴侣系统

基于 RAG（检索增强生成）技术的个性化智能学习助手，支持文档问答、知识图谱构建、自适应测验与苏格拉底式对话。

## 项目特色

- **权威问答** — 基于上传文档进行混合检索 + 重排序，引用原文生成回答，杜绝幻觉
- **苏格拉底式对话** — 不直接给答案，通过层层追问引导学生自主思考
- **智能测验** — 自动从文档生成题目，答错时分析原因并追问，强化薄弱知识点
- **知识图谱** — 自动抽取实体与关系，多文档对齐合并，交互式可视化
- **知识追踪** — 记录答题历史，计算掌握率，个性化出题权重
- **多格式支持** — PDF、Word、Markdown 文档解析（可选 LlamaParse 增强）
- **多模型路由** — 按用户 / 模块灵活切换 Google Gemini、Antigravity 反代等 LLM

## 快速开始

### 1. 环境准备

确保已安装 **Python 3.10+**（推荐 3.12）：

```bash
python --version
```

### 2. 创建虚拟环境

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / Mac
python -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置 API 密钥

复制环境变量模板并填写：

```bash
copy .env.example .env   # Windows
cp .env.example .env     # Linux / Mac
```

编辑 `.env` 文件，至少配置一种 LLM 调用方式：

```env
# ---- 方式 A：直连 Google Gemini（默认）----
GOOGLE_API_KEY=your_gemini_api_key

# ---- 方式 B：通过 Antigravity 反代（OpenAI 兼容）----
ANTIGRAVITY_API_KEY=your_antigravity_api_key
ANTIGRAVITY_API_BASE=https://your-domain/v1

# ---- 可选：增强功能 ----
LLAMA_CLOUD_API_KEY=your_llamaparse_key      # PDF 高质量解析
SILICONFLOW_API_KEY=your_siliconflow_key      # BGE-Reranker 重排序
```

**获取密钥：**

| 服务 | 地址 |
|------|------|
| Google Gemini | https://aistudio.google.com/ |
| Antigravity | Antigravity Tools 控制台 |
| LlamaParse | https://cloud.llamaindex.ai/ |
| 硅基流动 | https://siliconflow.cn/ |

> 若仅配置了 Antigravity（未设 `GOOGLE_API_KEY`），需在应用 **设置 → 模型选择** 中将各模块厂商切换为 Antigravity，并填写对应模型 ID。

### 5. 启动应用

**一键启动（推荐）：**

```bash
# Windows CMD
run.bat

# PowerShell
.\run.ps1
```

**手动启动：**

```bash
streamlit run app/主页.py
```

访问 http://localhost:8501 即可使用。

## 项目结构

```
bishe/
├── app/                          # Streamlit 前端
│   ├── 主页.py                    # 应用入口（登录 / 注册 / 首页）
│   ├── auth_cookie.py             # 登录状态持久化（Cookie + HMAC 签名）
│   └── pages/
│       ├── 1_智能问答.py           # RAG 问答 & 苏格拉底式对话
│       ├── 2_文档管理.py           # 文档上传 / 解析 / 管理
│       ├── 3_测验练习.py           # 自适应测验 & 知识追踪
│       ├── 4_知识图谱.py           # 交互式知识图谱可视化
│       └── 5_设置.py              # 模型选择 / 系统配置
├── backend/                       # 后端核心
│   ├── config.py                  # Pydantic-Settings 全局配置
│   ├── auth/
│   │   └── auth_service.py        # 注册 / 登录 / bcrypt 密码哈希
│   ├── database/
│   │   ├── models.py              # SQLAlchemy 数据模型
│   │   └── crud.py                # 数据库 CRUD 操作
│   ├── document/
│   │   ├── parser.py              # 多格式文档解析（PDF / Word / Markdown）
│   │   ├── chunker.py             # 父子分块 & 滑动窗口切分
│   │   └── embedder.py            # 向量嵌入（Gemini / OpenAI 兼容）
│   ├── retrieval/
│   │   ├── vector_store.py        # ChromaDB 向量存储
│   │   ├── hybrid_retriever.py    # 向量 + BM25 混合检索 & RRF 融合
│   │   └── reranker.py            # 自适应重排序（BGE-Reranker）
│   ├── rag/
│   │   ├── chain.py               # RAG 问答链
│   │   ├── prompts.py             # Prompt 模板
│   │   └── memory.py              # 对话记忆管理
│   ├── learning/
│   │   ├── quiz_generator.py      # 测验题目生成
│   │   ├── knowledge_tracker.py   # 知识掌握度追踪
│   │   ├── socratic_engine.py     # 苏格拉底式引导对话
│   │   └── kg_builder.py          # 知识图谱构建（实体抽取 / 对齐合并）
│   └── llm/
│       └── router.py              # 多厂商 LLM 路由
├── data/                          # 运行时数据（自动生成）
│   ├── learning_companion.db      # SQLite 数据库
│   ├── chroma_db/                 # ChromaDB 向量库
│   └── sample_docs/               # 示例文档
├── 教学文档/                       # 项目教学文档（6 篇）
├── .env.example                   # 环境变量模板
├── requirements.txt               # Python 依赖
├── run.bat                        # Windows CMD 启动脚本
├── run.ps1                        # PowerShell 启动脚本
└── README.md
```

## 技术栈

| 组件 | 技术 |
|------|------|
| 前端 | Streamlit |
| LLM | Google Gemini / Antigravity 反代（按模块可配置） |
| Embedding | gemini-embedding-001（可切换 OpenAI 兼容） |
| 向量数据库 | ChromaDB |
| RAG 框架 | LangChain + LlamaIndex |
| 文档解析 | LlamaParse / PyPDF / pdfplumber / python-docx |
| 重排序 | BGE-Reranker（硅基流动 API） |
| 知识图谱 | LLM 实体抽取 + 语义对齐合并 + streamlit-agraph 可视化 |
| 关系数据库 | SQLite + SQLAlchemy |
| 可视化 | Plotly |

## 核心功能

### RAG 问答流程

```
用户提问 → 混合检索（向量 + BM25）→ RRF 融合 → 自适应重排序 → 父文档上下文扩展 → LLM 生成 → 引用标注
```

### 苏格拉底式对话

```
用户提问 → 识别核心概念 → 生成引导性追问 → 用户作答 → 逐步深入 → 最终总结
```

### 智能测验

```
选择文档 → 检索核心知识 → LLM 生成题目 → 用户答题 → 错误分析追问 → 知识状态更新
```

### 知识图谱构建

```
文档分块 → 语义合并 → LLM 实体关系抽取 → 跨文档实体对齐 → 子图合并 → 交互式可视化
```

### 知识追踪

- 记录每个知识点的答题表现
- 掌握率 = 正确次数 / 总尝试次数
- 掌握率 < 60% 且尝试 ≥ 3 次 → 标记为薄弱点
- 出题时按权重采样，优先考察薄弱知识点

## 配置说明

系统采用 **双层配置**：

| 层级 | 存储位置 | 生效方式 | 内容 |
|------|---------|---------|------|
| `.env` 环境变量 | 项目根目录 `.env` 文件 | 重启后生效 | API 密钥、全局默认模型、RAG 参数 |
| 界面设置 | 数据库 `user_model_preferences` | 保存后立即生效 | 每个用户 / 模块的厂商、模型、temperature |

界面设置会覆盖 `.env` 默认值；未设置时回退到 `.env` 配置。

## 安全说明

- 用户密码使用 bcrypt 加密存储
- API 密钥仅存储在本地 `.env` 文件，不会上传
- 登录状态通过 HMAC 签名的 Cookie 维持
- 每个用户的文档和知识库相互隔离

## 许可

MIT License
