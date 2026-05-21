# rag_qa_chat.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
from pathlib import Path
from typing import List,Dict,Optional
import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import warnings
warnings.filterwarnings("ignore")
class Config:
    chroma_db_dir=r"F:\Pycharm\RAG\chroma_db_with_title_with_publish_date"
    model_dir=r"F:\Pycharm\RAG\models\Qwen\Qwen3-4B-Instruct-2507"
    collection_name="langchain"
    embedding_model_name="BAAI/bge-small-zh-v1.5"
    top_k=5
    max_new_tokens=512
    temperature=0.3
config=Config()

# ==================== 1. 加载向量数据库 ====================
def load_vector_db():
    """加载Chroma向量数据库"""
    print("正在加载向量数据库...")

    #创建一个"文字转向量"的翻译器
    embedding_fn=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config.embedding_model_name,
        device='cuda'if torch.cuda.is_available() else 'cpu'
    )

    # 连接到本地的Chroma数据库文件
    client=chromadb.PersistentClient(
        path=config.chroma_db_dir
    )

    # 从数据库中取出指定的"文件夹"（collection）
    collection=client.get_collection(
        name=config.collection_name,
        embedding_function=embedding_fn,
    )
    print(f"向量数据库加载完成，共{collection.count()}条数据")
    return collection

# ==================== 2. 加载Qwen模型 ====================
def load_qwen_model():
    """加载Qwen模型"""
    print("\n正在加载Qwen模型...")
    #4bit量化
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    #  加载分词器
    tokenizer=AutoTokenizer.from_pretrained(
        config.model_dir,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token

    #加载模型
    model=AutoModelForCausalLM.from_pretrained(
        config.model_dir,
        quantization_config=quantization_config,
        trust_remote_code=True,
        device_map="auto"
    )
    print(f"模型加载完成，设备：{model.device}\n")
    return model,tokenizer

# ==================== 3. 检索并返回最相关的3个新闻 ====================
def retrieve_new(collection,query:str,top_k:int=3)->List[Dict]:
    """从向量数据库中检索相关新闻"""
    #检索并返回3个最相关的新闻
    results=collection.query(
        query_texts=[query],
        n_results=top_k
    )

# ========results格式========
    # results = {
    #     'ids': [['id1', 'id2', 'id3']],
    #     'documents': [['正文1', '正文2', '正文3']],
    #     'metadatas': [[
    #         {'title': '标题1', 'url': 'url1', 'publish_date': '日期1'},
    #         {'title': '标题2', 'url': 'url2', 'publish_date': '日期2'},
    #         {'title': '标题3', 'url': 'url3', 'publish_date': '日期3'}
    #     ]],
    #     'distances': [[0.23, 0.45, 0.67]],
    #     'embeddings': None
    # }
    #将返回的新闻转化为另一种方便观看的格式（可转可不转）
    retrieved=[]
    if results['documents'] and results['documents'][0]:
        for i in range(len(results['documents'][0])):
            metadata=results['metadatas'][0][i] if results['metadatas'] else {}
            retrieved.append({
                'content':results['documents'][0][i],
                'title':metadata.get('title','无标题'),
                'url':metadata.get('url','无URL'),
                'publish_date':metadata.get('publish_date','未知时间')
            })

# =========retrieved格式========
    # retrieved = [
    #     {
    #         'content': '新闻正文...',
    #         'title': '新闻标题',
    #         'url': 'https://...',
    #         'publish_date': '2024-08-16'
    #     },
    #     {
    #         'content': '新闻正文...',
    #         'title': '新闻标题',
    #         'url': 'https://...',
    #         'publish_date': '2024-08-16'
    #     }
    # ]
    return retrieved

# ==================== 4. 构建Prompt ====================
def build_prompt(query:str,retrieved_news:List[Dict])->tuple:
    """构建system和uer_prompt"""
    system_prompt = """
    你是一个专业的时间感知新闻问答助手，你的任务是基于提供的新闻内容回答用户问题。
    严格要求：
    1.只使用提供的新闻内容来回答问题，不要使用你自己的知识
    2.如果新闻中包含时间信息，请在答案中明确说明
    3.答案一定要简洁准确，不要说多余的废话，可以引用新闻中的原话
    4.如果相关新闻没有找到关于这个新闻的话你要明确说明：“根据现有新闻找不到该问题相关的信息”
    """

    context_parts=[]
    for i,new in enumerate(retrieved_news,1):
        context_parts.append(f"""
        【参考新闻：{i}】
        标题：{new['title']}   
        发布时间：{new['publish_date']}
        内容：{new['content']}
        """)
    context="\n".join(context_parts)

    user_prompt=f"""
    请根据以下新闻回答用户问题。
    新闻：{context}
    用户问题：{query}
    """
    return system_prompt,user_prompt

# ==================== 5. 生成答案 ====================
def generate_answer(model,tokenizer,query:str,retrieved_news:List[Dict])->str:
    """基于检索结果生成并返回答案"""
    system_prompt,user_prompt=build_prompt(query,retrieved_news)
    messages=[
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_prompt}
    ]

    text=tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True  #在末尾加上助手回答
    )

    inputs=tokenizer(
        [text],
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs=model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    response=tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    return response

# ==================== 6. 打印结果  ====================
def print_answer_with_urls(query:str,answer:str,retrieved_news:List[Dict]):
    """打印答案和网址来源"""
    print("\n"+"="*70)
    print(f"问题：{query}")
    print("="*70)

    print(f"\n答案：\n{answer}")

    # 打印信息来源
    print("\n"+"-"*50)
    print("新闻来源：")
    seen_urls=set()
    for i,news in enumerate(retrieved_news,1):
        url=news.get('url','')
        if url!='无URL' and url not in seen_urls:
            seen_urls.add(url)
            print(f"[{i}] {url}")
        elif not url or url=='无URL':
            print(f"[{i}] 无URL")
    print("\n"+"="*70)

# ==================== 7. 交互式问答 ====================
def interactive_qa(model,tokenizer,collection):
    """交互式问答"""

    print("\n" + "=" * 70)
    print("ChronoQA RAG 智能问答系统")
    print("=" * 70)
    print("输入'exit'退出")
    print("="*70)

    while True:
        user_input=input("\n请输入您的问题：").strip()
        if not user_input:
            continue

        if user_input.lower()=="exit":
            print("\n再见！")
            break

        # 开始回答问题
        print("正在检索新闻...")
        retrieved=retrieve_new(collection,user_input,config.top_k)
        if not retrieved:
            print("未能找到该问题相关的新闻!")

        print("正在生成答案...")
        answer=generate_answer(model,tokenizer,user_input,retrieved)
        print_answer_with_urls(user_input,answer,retrieved)

# ==================== 主函数 ====================
def main():
    print("正在启动ChronoQA RAG 系统...\n")
    collection=load_vector_db()
    model,tokenizer=load_qwen_model()
    interactive_qa(model,tokenizer,collection)


if __name__ == "__main__":
    main()