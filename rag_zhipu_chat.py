import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from pathlib import Path
from typing import List, Dict, Optional
import chromadb
from chromadb.utils import embedding_functions
from zhipuai import ZhipuAI
import warnings

warnings.filterwarnings("ignore")

class Config:
    chroma_db_dir = r"F:\Pycharm\RAG\chroma_db_with_title_with_publish_date"
    collection_name = "langchain"
    embedding_model_name = "BAAI/bge-small-zh-v1.5"
    top_k = 5
    max_new_tokens = 512
    temperature = 0.3

    # 智谱 API 配置
    zhipu_api_key = "e241e6d2788a4c5fb08b1f1ea2f0d1f1.0iJs0FM7vE5oesHc"  # 替换为你的实际 API Key
    zhipu_model = "glm-4-flash"  # 可选: glm-4-flash, glm-4-plus, glm-4-air


config = Config()


# ==================== 1. 加载向量数据库 ====================
def load_vector_db():
    """加载Chroma向量数据库"""
    print("正在加载向量数据库...")

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.embedding_model_name,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    client = chromadb.PersistentClient(path=config.chroma_db_dir)
    collection = client.get_collection(
        name=config.collection_name,
        embedding_function=embedding_fn,
    )
    print(f"向量数据库加载完成，共{collection.count()}条数据")
    return collection


# ==================== 2. 加载模型（智谱 API）====================
def load_qwen_model():
    """加载模型（实际是初始化智谱API客户端）"""
    print("\n正在加载智谱API...")

    client = ZhipuAI(api_key=config.zhipu_api_key)

    print(f"智谱API加载完成，使用模型：{config.zhipu_model}")
    return client, None  # 返回 client 和 None（保持接口一致）


# ==================== 3. 检索新闻 ====================
def retrieve_new(collection, query: str, top_k: int = 3) -> List[Dict]:
    """从向量数据库中检索相关新闻"""
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    retrieved = []
    if results['documents'] and results['documents'][0]:
        for i in range(len(results['documents'][0])):
            metadata = results['metadatas'][0][i] if results['metadatas'] else {}
            retrieved.append({
                'content': results['documents'][0][i],
                'title': metadata.get('title', '无标题'),
                'url': metadata.get('url', '无URL'),
                'publish_date': metadata.get('publish_date', '未知时间')
            })
    return retrieved


# ==================== 4. 构建 Prompt ====================
def build_prompt(query: str, retrieved_news: List[Dict]) -> tuple:
    """构建 system 和 user prompt"""
    system_prompt = """
    你是一个专业的时间感知新闻问答助手，你的任务是基于提供的新闻内容回答用户问题。
    严格要求：
    1. 只使用提供的新闻内容来回答问题，不要使用你自己的知识
    2. 如果新闻中包含时间信息，请在答案中明确说明
    3. 答案一定要简洁准确，不要说多余的废话，可以引用新闻中的原话
    4. 如果相关新闻没有找到关于这个问题的信息，你要明确说明："根据现有新闻找不到该问题相关的信息"
    """

    context_parts = []
    for i, news in enumerate(retrieved_news, 1):
        context_parts.append(f"""
        【参考新闻 {i}】
        标题：{news['title']}
        发布时间：{news['publish_date']}
        内容：{news['content']}
        """)
    context = "\n".join(context_parts)

    user_prompt = f"""
    请根据以下新闻回答用户问题。
    新闻：{context}
    用户问题：{query}
    """
    return system_prompt, user_prompt


# ==================== 5. 生成答案（智谱 API）====================
def generate_answer(model, tokenizer, query: str, retrieved_news: List[Dict]) -> str:
    """基于检索结果生成答案（使用智谱API）"""
    client = model  # model 是 ZhipuAI 客户端

    system_prompt, user_prompt = build_prompt(query, retrieved_news)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = client.chat.completions.create(
            model=config.zhipu_model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_new_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"智谱API调用失败: {e}")
        return "生成答案失败"


# ==================== 6. 打印结果 ====================
def print_answer_with_urls(query: str, answer: str, retrieved_news: List[Dict]):
    """打印答案和网址来源"""
    print("\n" + "=" * 70)
    print(f"问题：{query}")
    print("=" * 70)

    print(f"\n答案：\n{answer}")

    print("\n" + "-" * 50)
    print("新闻来源：")
    seen_urls = set()
    for i, news in enumerate(retrieved_news, 1):
        url = news.get('url', '')
        if url != '无URL' and url not in seen_urls:
            seen_urls.add(url)
            print(f"[{i}] {url}")
        elif not url or url == '无URL':
            print(f"[{i}] 无URL")
    print("\n" + "=" * 70)


# ==================== 7. 交互式问答 ====================
def interactive_qa(model, tokenizer, collection):
    """交互式问答"""
    print("\n" + "=" * 70)
    print("ChronoQA RAG 智能问答系统（智谱API版）")
    print("=" * 70)
    print("输入'exit'退出")
    print("=" * 70)

    while True:
        user_input = input("\n请输入您的问题：").strip()
        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("\n再见！")
            break

        print("正在检索新闻...")
        retrieved = retrieve_new(collection, user_input, config.top_k)
        if not retrieved:
            print("未能找到该问题相关的新闻!")

        print("正在生成答案...")
        answer = generate_answer(model, None, user_input, retrieved)
        print_answer_with_urls(user_input, answer, retrieved)


# ==================== 主函数 ====================
def main():
    print("正在启动ChronoQA RAG 系统...\n")
    collection = load_vector_db()
    model, tokenizer = load_qwen_model()
    interactive_qa(model, tokenizer, collection)


if __name__ == "__main__":
    main()