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
| Qwen3-4B（本地） | 49.48% | 65.74% |
| 智谱 GLM-4-Flash（API） | 58.48% | 65.74% |

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

## 2. 配置国内镜像（解决国内连接不到导致下载不了模型的问题）
#在代码开头加上
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
