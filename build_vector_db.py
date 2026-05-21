import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# LangChain相关导入
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
# ==================== 1.配置 ====================
cleaned_news_dir=Path(r"F:\Pycharm\RAG\news_corpus_simple\cleaned")
vector_db_dir=Path(r"F:\Pycharm\RAG\chroma_db") #向量数据库保存目录

chunk_size=700  #每块的最大字符
chunk_overlap=50

Embedding_model="BAAI/bge-small-zh-v1.5"    #Embedding模型配置
batch_size=1000  #批处理大小

# ==================== 2.加载文件 ====================
def load_json_files(data_dir:Path)->List[Dict]:
    """加载所有cleaned的json文件"""
    json_files=list(data_dir.glob("*.json"))#在目录中加载所有以.json结尾的文件
    print(f"找到{len(json_files)}个json文件")
    all_data=[]
    for file_path in json_files:
        try:
            with open(file_path,'r',encoding='utf-8') as f:
                data=json.load(f)
                if data.get('content') and len(data['content'])>50:
                    all_data.append(data)
        except Exception as e:
            print(f"读取失败：{e}")
    print(f"有效新闻数：{len(all_data)}")
    return all_data

# ==================== 3.内容分块 ====================
def split_documents(news_list:List[Dict])->List[Document]:
    """将新闻内容分成小块"""
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
    )
    documents=[]
    for news in news_list:
        content=news['content']
        if not content:
            continue

        chunks=text_splitter.split_text(content)    #切分内容

        for i,chunk in enumerate(chunks):
            doc=Document(
                page_content=chunk,
                metadata={
                    'title': news.get('title',''),
                    'url':news.get('url',''),
                    'publish_date': news.get('publish_date',''),
                    'qa_date':news.get('qa_date',''),
                    'chunk_index':i,
                }
            )
            documents.append(doc)

    print(f"切分后共{len(documents)}个文本块")
    print(f"平均每个新闻切分成 {len(documents) / len(news_list):.1f} 块")
    return documents

# ==================== 4.构建向量数据库 ====================
def build_vector_db(documents:List[Document],persist_dir:Path):
    """构建向量数据库"""
    print("\n正在加载Embedding模型...")
    print(f"模型：{Embedding_model}")

    #加载Embedding模型
    embeddings=HuggingFaceEmbeddings(
        model_name=Embedding_model,
        model_kwargs={'device':'cuda'},
        encode_kwargs={'normalize_embeddings':True}
    )
    print("Embedding模型加载完成")
    print(f"开始向量化并存入数据库，共{len(documents)}个文档块...")

    #创建向量数据库
    if len(documents)>batch_size:   #第一批创建数据库
        print(f"处理第1/{len(documents)//batch_size+1}批...")
        vector_db=Chroma.from_documents(
            documents=documents[:batch_size],
            embedding=embeddings,
            persist_directory=str(persist_dir),
        )
        #后续批次添加
        for i in range(batch_size,len(documents),batch_size):
            batch=documents[i:i+batch_size]
            vector_db.add_documents(batch)
            print(f"处理第{i//batch_size+1}批，"
                  f"已添加{min(i+batch_size,len(documents))}/{len(documents)}")
    else:
        vector_db=Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=str(persist_dir)
        )
    print(f"\n向量数据库已保存到：{persist_dir}")
    return vector_db

# ==================== 5.测试检索功能 ====================
def test_retrieval(vector_db,query:str,k:int=3):
    """测试检索功能"""
    print(f"\n测试检索：'{query}'")
    results=vector_db.similarity_search(query,k=k)
    for i,doc in enumerate(results):
        print(f"\n结果{i+1}:")
        print(f"标题：{doc.metadata.get('title','无')}")
        print(f"网址：{doc.metadata.get('url','无')}")
        print(f"内容预览：{doc.page_content[:100]}...")

def main():
    print("="*50)
    print("构建RAG向量知识库")
    print("="*50)

    print("\n1.加载清洗后的新闻文件...")
    news_list=load_json_files(cleaned_news_dir)
    if len(news_list)==0:
        print("加载的新闻文件不存在")
        return

    print("\n2.文本切割...")
    documents=split_documents(news_list)
    if len(documents)==0:
        print("文本切割后没有生成文档块")
        return

    print("3.构建向量数据库...")
    vector_db=build_vector_db(documents,vector_db_dir)

    print("4.测试检索功能...")
    test_retrieval(vector_db,"沧州雄狮 vs 河南 比赛结果")
    test_retrieval(vector_db,"小米汽车即将上市,雷军回应“价格战”")

    print("\n"+"="*50)
    print(f"向量数据库构建完成，已保存到{vector_db_dir}")

if __name__ == "__main__":
    main()