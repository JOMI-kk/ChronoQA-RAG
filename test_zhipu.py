#test_zhipu.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pandas as pd
import ast
import re
import json
from typing import List, Dict
from datetime import datetime
import torch
import chromadb
from chromadb.utils import embedding_functions
from zhipuai import ZhipuAI
import warnings
warnings.filterwarnings("ignore")
from test_rag import get_golden_urls,is_url_match,is_answer_correct
from rag_qa_chat import load_vector_db,retrieve_new,build_prompt


# ==================== 配置 ====================
class Config:
    # 向量数据库配置
    chroma_db_dir = r"F:\Pycharm\RAG\chroma_db_with_title_with_publish_date"
    collection_name = "langchain"
    embedding_model_name = "BAAI/bge-small-zh-v1.5"
    top_k = 5

    # 智谱 API 配置
    zhipu_api_key = "e241e6d2788a4c5fb08b1f1ea2f0d1f1.0iJs0FM7vE5oesHc"
    zhipu_model = "glm-4-flash"
    max_new_tokens = 512
    temperature = 0.3

    # 测试配置
    chronoqa_path = r"F:\下载\ChronoQA-main\chronoqa.csv"

config = Config()

# ==================== 初始化智谱客户端 ====================
def init_zhipu_client():
    """初始化智谱API客户端"""
    print("\n正在加载智谱API...")
    client=ZhipuAI(api_key=config.zhipu_api_key)
    print(f"智谱API加载完成！，使用模型：{config.zhipu_model}")
    return client

# ==================== 生成智谱答案 ====================
def generate_answer(client,query:str,retrieved_news:List[Dict])->str:
    """基于检索结果生成答案"""
    system_prompt,user_prompt=build_prompt(query,retrieved_news)
    messages=[
        {"role":"system","content":system_prompt},
        {"role":"user","content":user_prompt}
    ]
    try:
        response=client.chat.completions.create(
            model=config.zhipu_model,
            messages=messages,
            temperature=config.temperature,
            max_tokens=config.max_new_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"智谱API调用失败：{e}")
        return "生成答案失败"

# ==================== 测试 ====================
def main():
    """运行批量测试"""
    print('=' * 70)
    print(f"开始测试 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('=' * 70)


    # ============（1）.加载向量数据库==========
    print("\n正在加载向量数据库...")
    collection=load_vector_db()

    # ============（2）.初始化智谱客户端==========
    client=init_zhipu_client()

    # ===========(3)加载问答数据==============
    print("\n正在加载测试数据...")
    df=pd.read_csv(config.chronoqa_path)
    total_sample=len(df)

    answer_correct_list = []
    url_match_list=[]
    start_time=datetime.now()

    # 逐行提取全部数据，自定义i从1开始
    for i,(idx,row) in enumerate(df.iterrows(),1):
        question=row["question"]
        golden=str(row['answer'])
        golden_urls=get_golden_urls(row)

        #显示进度
        print(f"\n{'='*70}")
        print(f"进度：{i}/{total_sample}")
        print(f"问题：{question}")
        print(f"标准答案：{golden}")
        print(f"标准答案URL:{golden_urls[0]}")

        #============(4)检索得到最相关的topo_k个新闻==========
        retrieved=retrieve_new(collection,question,config.top_k)

        #============(5)判断是否能检索到正确的URL,判断答案是否正确===========
        predicted=generate_answer(client,question,retrieved)
        url_match=is_url_match(retrieved,golden_urls)
        answer_correct=is_answer_correct(golden,predicted)

        answer_correct_list.append(answer_correct)
        url_match_list.append(url_match)

        print(f"模型答案：{predicted}")
        print(f"URL是否匹配：{'✅' if url_match else '❌'}")
        print(f"答案是否正确：{'✅'if answer_correct else '❌'}")

        print(f"检索到的URL(Top5):")
        for j ,new in enumerate(retrieved[:5],1):
            url=new.get('url','')
            print(f"[{j}]{url}")

    # =============(6)统计信息=============
    elapsed_time=((datetime.now()-start_time).total_seconds())/60
    answer_accuracy=sum(answer_correct_list)/total_sample*100
    url_accuracy=sum(url_match_list)/total_sample*100
    print(f"\n{'='*70}")
    print("测试完成！")
    print(f"总耗时：{elapsed_time:.2f}分钟")
    print(f"答案准确性：{sum(answer_correct_list)}/{total_sample} = {answer_accuracy:.2f}%")
    print(f"URL检索正确率：{sum(url_match_list)}/{total_sample} = {url_accuracy:.2f}%")
    print(f"{'=' * 70}")

    # =============(7)保存结果=============
    results_dir=r"zhipu_test_results"
    os.makedirs(results_dir,exist_ok=True)

    # 保存为JSON
    timestamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    result = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "total_samples": total_sample,
        "answer_accuracy": round(answer_accuracy, 2),
        "url_accuracy": round(url_accuracy, 2),
        "correct_answers": sum(answer_correct_list),
        "correct_urls": sum(url_match_list),
        "elapsed_minutes": round(elapsed_time, 2)
    }

    with open(os.path.join(results_dir,f"result_{timestamp}.json"),'w',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False, indent=2)

    print(f"\n结果已保存至：{results_dir}\\result_{timestamp}.json")

# ==================== 主函数 ====================
if __name__ == "__main__":
    main()