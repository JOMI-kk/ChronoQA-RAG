# ChronoQA-RAG：时间感知的新闻问答系统

基于 RAG（Retrieval-Augmented Generation）架构的时间敏感型新闻问答系统。使用 ChronoQA 数据集，结合本地 Qwen3-4B 模型和智谱 API，实现基于新闻的时间敏感问答。

## 🎯 项目目标

构建一个能够**理解时间信息**的智能问答系统：

- 用户提问：“COTODAMA歌词音箱和苹果停产iPhone 6系列哪个事件更早发生？”
- 系统检索相关新闻 → 理解时间关系 → 生成答案：“COTODAMA歌词音箱更早”
- 同时输出答案来源（新闻URL），保证可追溯
<img width="1118" height="422" alt="image" src="https://github.com/user-attachments/assets/10ffe7d5-383a-4b93-ab01-009456774df5" />

## 📊 项目结果

## 测试结果（5176个问答对样本）

| 模型 | 答案正确率 | URL检索正确率 |
|------|-----------|--------------|
| Qwen3-4B（本地） | （2561）49.48% | （3396）65.74% |
| 智谱 GLM-4-Flash（API） |（3026） 58.48% | （3396）65.74% |

## 📁 项目结构
```text
RAG/
├── chroma_db/                                    # 向量数据库（旧版，只有content作为page_content，其他的键作为metadata）
├── chroma_db_with_title_with_publish_date/       # 向量数据库（增强版，content+title+publish_date作为page_content）
├── models/                                       # 本地 Qwen 模型
│   └── Qwen/
│       └── Qwen3-4B-Instruct-2507/              # Qwen3-4B 模型文件
├── news_corpus_simple/                          # 原始新闻数据（6060篇新闻）
│   └── cleaned/                                  # 清洗后的新闻 JSON 文件，共5034篇新闻作为原始数据库（过滤掉1026篇无效URL、404页面的新闻）
├── qwen_test_results/                           # Qwen 模型测试结果
├── zhipu_test_results/                          # 智谱 API 测试结果
├── build_vector_db.py                           # 构建向量数据库
├── build_vector_db_new.py                       # 构建向量数据库（增强版）
├── check_data.py                                # 查看数据集信息
├── clean_and_filter.py                          # 清洗新闻数据
├── download_qwen_model.py                       # 下载 Qwen 模型
├── get_original_data.py                         # 爬取原始新闻
├── rag_qa_chat.py                               # RAG交互式问答（本地 Qwen）
├── rag_zhipu_chat.py                            # RAG交互式问答（智谱 API）
├── test_rag.py                                  # 本地 Qwen 测试（基于5034篇新闻，测试5176个问答对，统计答案准确率和 URL 检索召回率）
└── test_zhipu.py                                # 智谱 API 测试（基于5034篇新闻，测试5176个问答对，统计答案准确率和 URL 检索召回率）
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 创建 conda 环境
conda create -n ChronoQA-RAG python=3.10
conda activate ChronoQA-RAG

# 安装依赖
pip install -r requirements.txt
torch>=2.0.0
transformers>=4.37.0
sentence-transformers>=2.2.0
chromadb>=0.4.0
pandas>=2.0.0
zhipuai>=2.0.0
bitsandbytes>=0.41.0
trafilatura>=1.6.0
jieba>=0.42.0
rank-bm25>=0.2.0
```
### 2. 配置国内镜像（解决国内网络连接不到 HuggingFace 导致下载模型失败的问题）
```bash
#在代码开头加上
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

###  3. 准备数据
```bash
# 爬取新闻数据
python get_original_data.py

# 检查数据集信息
python check_data.py

# 清洗新闻数据
python clean_and_filter.py
```
###  4. 构建向量数据库
```bash
# 构建向量数据库（含正文、标题、发布时间）
python build_vector_db_new.py
```
###  5. 下载 Qwen 模型（本地方案）
```bash
python download_qwen_model.py
```
### 6.运行问答系统
```bash
# 交互式问答（本地 Qwen）
python rag_qa_chat.py

# 交互式问答（智谱 API）
python rag_zhipu_chat.py
```
### 7.测试准确率
```bash
# 测试本地 Qwen
python test_rag.py

# 测试智谱 API
python test_zhipu.py
```
## 📖 各脚本功能说明

| 脚本 | 功能 | 输出 |
|------|------|------|
| `get_original_data.py` | 从 ChronoQA 数据集爬取新闻原文 | `news_corpus_simple/articles/` |
| `clean_and_filter.py` | 清洗新闻（去广告、过滤404、过滤无效URL） | `news_corpus_simple/cleaned/` |
| `check_data.py` | 查看数据集统计信息 | 控制台输出 |
| `build_vector_db_new.py` | 构建向量数据库（标题+正文+发布时间联合向量化） | `chroma_db_with_title_with_publish_date/` |
| `download_qwen_model.py` | 下载 Qwen3-4B 模型（4bit量化） | `models/Qwen/` |
| `rag_qa_chat.py` | 本地 Qwen 模型交互式问答 | 控制台交互 |
| `rag_zhipu_chat.py` | 智谱 API 交互式问答 | 控制台交互 |
| `test_rag.py` | 批量测试本地 Qwen 模型 | `qwen_test_results/result_20260518_181304.json` |
| `test_zhipu.py` | 批量测试智谱 API | `zhipu_test_results/test_zhipu_20260520_221638.json` |



## ⚙️ 核心配置

### 向量数据库配置（rag_qa_chat.py）

```python
class Config:
    chroma_db_dir = r"F:\Pycharm\RAG\chroma_db_with_title_with_publish_date"
    collection_name = "langchain"
    embedding_model_name = "BAAI/bge-small-zh-v1.5"
    top_k = 5
    max_new_tokens = 512
    temperature = 0.3
```

### 智谱 API 配置（rag_zhipu_chat.py）

```python
class Config:
    zhipu_api_key = "e241e6d2788a4c5fb08b1f1ea2f0d1f1.0iJs0FM7vE5oesHca"  
    zhipu_model = "glm-4-flash"     
    max_new_tokens = 512
    temperature = 0.3
```


## ❓ 常见问题与解决

### 1. 网络问题（HuggingFace 连接超时）

```python
# 在代码开头添加
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
```

### 2. 显存不足（8G以下）

```python
# 使用 4bit 量化加载模型
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
)
```
### 3. LangChain 版本适配
LangChain 1.0 版本对模块结构进行了重构，旧版导入路径已变更。

**旧版导入（已废弃）：**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
```

**新版本导入：**
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
```
### 4.向量数据库的检索质量不佳，用户问题与目标新闻的语义匹配度低，导致正确答案对应的 URL 无法被成功召回。

（1）**向量数据库优化**：原版本仅将 content 作为检索内容，增强版将 content + title + publish_date 共同作为 page_content 参与向量化，提升检索召回率。

（2）**检索数量调整**：将 top_k从 3 增加到 5，返回更多候选新闻，提高正确答案命中率。

### 5.检索到正确 URL 后，模型推理能力不足，无法基于新闻内容生成正确答案。

解决方案：

（1）**模型升级**：将本地 Qwen3-4B 模型更换为智谱 GLM-4-Flash 模型（免费 API），提升答案生成的准确率。

（2）**字符串精确匹配**：直接比较标准答案与模型答案是否完全一致或存在包含关系

（3） **数字匹配**：提取答案中的数字进行比对，解决数值类答案格式差异问题

（4）**中文数字转换匹配**：支持中文数字（一、二、三）与阿拉伯数字（1、2、3）的互转匹配

（5） **关键词匹配**：提取答案中的中文关键词（2字及以上），计算重合比例达到25%即判定为正确

### 6.**数据集局限性：**

1. **数据来源**：ChronoQA 数据集来源于新浪新闻，其中的问答对由模型自动生成，经过我个人验证，小部分答案存在不准确的情况。

2. **URL 失效问题**：原始新闻共 6,060 篇，其中 1,026 篇（约 17%）的 URL 已失效（返回 404），无法获取原文。这导致 RAG 系统在检索时无法命中这些失效链接，从而降低了 URL 召回率。
<img width="503" height="686" alt="image" src="https://github.com/user-attachments/assets/d395f9ea-7256-454f-9bf0-f49874ec0056" />

<img width="519" height="1193" alt="image" src="https://github.com/user-attachments/assets/cee86d16-6caf-47ac-88ed-2b4a570f0d2a" />

