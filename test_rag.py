# test_rag.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pandas as pd
import ast
import re
import sys
from typing import List, Dict, Set
import json
from datetime import datetime
sys.path.append(r"F:\Pycharm\RAG")
from rag_qa_chat import load_vector_db,load_qwen_model,retrieve_new,generate_answer

chronoqa_path = r"F:\下载\ChronoQA-main\chronoqa.csv"
top_k=5

#=============获取标准答案的URL============

def get_golden_urls(row)->List[str]:
    """获取标准答案的URL"""
    try:
        urls_str=row.get('golden_chunks_urls','')
        if pd.isna(urls_str) or urls_str=='':
            return []
        return ast.literal_eval(urls_str)
    except:
        return []

# ============检查检索到的URL是否包含标准答案的URL=============

def is_url_match(retrieved_news:List[Dict],golden_urls:List[str])->bool:
    """检查检索到的URL是否包含标准答案的URL"""
    if not golden_urls:
        return False
    retrieved_urls={news.get('url','') for news in retrieved_news if news.get('url')}
    golden_urls_set=set(golden_urls)

    return bool(retrieved_urls & golden_urls_set)

# ==============判断答案是否正确=============

def is_answer_correct(golden:str,predicted:str)->bool:
    """判断答案是否正确"""
    if not predicted or "无法回答" in predicted:
        return False
    golden=str(golden).strip()
    predicted=predicted.strip()

    #1.字符串精确匹配
    if golden==predicted or (golden in predicted) or (predicted in golden):
        return True

    #2.数字匹配
    nums_golden=re.findall(r'\d+',golden)#获取golden的数字，+表示多个
    nums_pred=re.findall(r'\d+',predicted)
    if nums_golden and nums_pred and nums_golden[0]== nums_pred[0]:
        return True

    #3.中文数字转换匹配
    chinese_nums={'一':'1','二':'2','三':'3','四':'4','五':'5','六':'6','七':'7','八':'8','九':'9','十':'10'}
    for ch,num in chinese_nums.items():
        if ch in golden and num in predicted:
            return True
        if num in golden and ch in predicted:
            return True

    #4.关键字匹配,得到去重的长度至少为2中文词语组
    keywords_golden=set(re.findall(r'[\u4e00-\u9fa5]{2,}', golden))
    keywords_pred = set(re.findall(r'[\u4e00-\u9fa5]{2,}', predicted))

    if keywords_golden and keywords_pred:
       intersection=keywords_golden & keywords_pred
       ratio=len(intersection)/len(keywords_golden)

       if ratio>=0.25:  #如果模型回答的答案的词语组占标准答案的0.25就说明这个答案正确
           return True
    return False

def main():
    print('='*70)
    print(f"开始测试-{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*70)

    print("\n正在加载数据...")
    collection=load_vector_db()
    model,tokenizer=load_qwen_model()

    #加载问答对的数据
    df=pd.read_csv(chronoqa_path)
    test_samples=df.sample(n=934,random_state=11)#.sample()表示随机抽样
    total_samples=len(test_samples)

    answer_correct_list=[]
    url_match_list=[]
    start_time=datetime.now()

    #逐行提取test_sample的数据，自定义current_num从1开始
    for current_num,(idx,row) in enumerate(test_samples.iterrows(),1):
        question=row['question']
        golden=str(row['answer'])
        golden_urls=get_golden_urls(row)

        #显示进度
        print(f"\n{'='*70}")
        print(f"进度：{current_num}/{total_samples}")
        print(f"问题：{question} ")
        print(f"标注答案：{golden} ")
        if golden_urls:
            print(f"标准答案URL:{golden_urls[0]}")
        retrieved=retrieve_new(collection,question,top_k)#得到最相关的top_k个新闻

        if not retrieved:
            predicted="未找到相关新闻"
            url_match=False
            answer_correct=False
        else:
            predicted=generate_answer(model,tokenizer,question,retrieved)#得到模型的回答
            url_match=is_url_match(retrieved,golden_urls)#判断检索到的url和标准答案的url是否相等
            answer_correct=is_answer_correct(golden,predicted)#判断模型回答的答案是否正确

        answer_correct_list.append(answer_correct)
        url_match_list.append(url_match)

        print(f"模型答案：{predicted} ")
        print(f"URL匹配：{'✅' if url_match else '❌'}")
        print(f"答案正确: {'✅' if answer_correct else '❌'}")

        if retrieved:
            print(f"\n检索到的URL(Top3):")
            for j,new in enumerate(retrieved[:3],1):
                url=new.get('url','无URL')
                print(f"[{j}]{url}")

    #计算统计信息
    elapsed_time=((datetime.now()-start_time).total_seconds())/60
    answer_accuracy=sum(answer_correct_list)/total_samples*100
    url_accuracy=sum(url_match_list)/total_samples*100

    print(f"\n{'='*70}")
    print("测试完成！")
    print(f"总耗时：{elapsed_time:.1f}分钟")
    print(f"答案正确率： {sum(answer_correct_list)}/{total_samples} = {answer_accuracy:.2f}%")
    print(f"URL检索正确率：{sum(url_match_list)}/{total_samples} = {url_accuracy:.2f}%")
    print(f"{'=' * 70}")

    # ========== 保存结果 ==========
    # 创建结果目录
    results_dir = r"/qwen_test_results"
    os.makedirs(results_dir, exist_ok=True)

    # 保存为 JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "total_samples": total_samples,
        "answer_accuracy": round(answer_accuracy, 2),
        "url_accuracy": round(url_accuracy, 2),
        "correct_answers": sum(answer_correct_list),
        "correct_urls": sum(url_match_list),
        "elapsed_minutes": round(elapsed_time, 2)
    }

    with open(os.path.join(results_dir, f"result_{timestamp}.json"), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 结果已保存至: {results_dir}\\result_{timestamp}.json")

if __name__ == '__main__':
    main()