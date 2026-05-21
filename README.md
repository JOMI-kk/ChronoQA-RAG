# ChronoQA-RAG：时间感知的新闻问答系统

基于 RAG（Retrieval-Augmented Generation）架构的时间敏感型新闻问答系统。使用 ChronoQA 数据集，结合本地 Qwen3-4B 模型和智谱 API，实现基于新闻的时间敏感问答。

## 🎯 项目目标

构建一个能够**理解时间信息**的智能问答系统：

- 用户提问：“COTODAMA歌词音箱和苹果停产iPhone 6系列哪个事件更早发生？”
- 系统检索相关新闻 → 理解时间关系 → 生成答案：“COTODAMA歌词音箱更早”
- 同时输出答案来源（新闻URL），保证可追溯
<img width="1118" height="422" alt="image" src="https://github.com/user-attachments/assets/10ffe7d5-383a-4b93-ab01-009456774df5" />

## 📊 项目结果

## 测试结果（5176个样本）

| 模型 | 答案正确率 | URL检索正确率 |
|------|-----------|--------------|
| Qwen3-4B（本地） | 49.48% | 65.74% |
| 智谱 GLM-4-Flash（API） | 58.48% | 65.74% |

## 📁 项目结构
RAG/
├── chroma_db/ # 向量数据库（旧版）
├── chroma_db_with_title_with_publish_date/ # 向量数据库（新版，含元数据）
├── models/ # 本地 Qwen 模型
│ └── Qwen/
│ └── Qwen3-4B-Instruct-2507/ # Qwen3-4B 模型文件
├── news_corpus_simple/ # 原始新闻数据
│ └── cleaned/ # 清洗后的新闻 JSON 文件
├── qwen_test_results/ # Qwen 模型测试结果
├── zhipu_test_results/ # 智谱 API 测试结果
│
├── build_vector_db.py # 构建向量数据库
├── build_vector_db_new.py # 构建向量数据库（新版，含标题/时间）
├── check_data.py # 查看数据集信息
├── clean_and_filter.py # 清洗新闻数据
├── download_qwen_model.py # 下载 Qwen 模型
├── get_original_data.py # 爬取原始新闻
├── rag_qa_chat.py # RAG 问答主程序（本地 Qwen）
├── rag_zhipu_chat.py # RAG 问答主程序（智谱 API）
├── test_rag.py # 本地 Qwen 测试脚本
├── test_zhipu.py # 智谱 API 测试脚本
├── documents_cache.pkl # BM25 索引缓存
├── qwen3_model_path.txt # Qwen 模型路径记录
└── 150/ # 测试数据目录
