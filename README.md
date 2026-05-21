# ChronoQA-RAG：时间感知的新闻问答系统

基于 RAG（Retrieval-Augmented Generation）架构的时间敏感型新闻问答系统。使用 ChronoQA 数据集，结合本地 Qwen3-4B 模型和智谱 API，实现基于新闻的时间敏感问答。

## 🎯 项目目标

构建一个能够**理解时间信息**的智能问答系统：

- 用户提问：“COTODAMA歌词音箱和苹果停产iPhone 6系列哪个事件更早发生？”
- 系统检索相关新闻 → 理解时间关系 → 生成答案：“COTODAMA歌词音箱更早”
- 同时输出答案来源（新闻URL），保证可追溯
<img width="1118" height="422" alt="image" src="https://github.com/user-attachments/assets/10ffe7d5-383a-4b93-ab01-009456774df5" />
