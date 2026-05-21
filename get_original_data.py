import pandas as pd
import requests
import time
import json
import hashlib
from pathlib import Path
from bs4 import BeautifulSoup
import trafilatura
import re

# 正确的路径
data_dir = r"F:\下载\ChronoQA-main"
csv_path = r"F:\下载\ChronoQA-main\chronoqa.csv"
output_dir = Path("news_corpus_simple")
output_dir.mkdir(exist_ok=True)

# 创建文章保存目录
articles_dir = output_dir / "articles"
articles_dir.mkdir(exist_ok=True)

print(f"数据目录: {data_dir}")
print(f"CSV文件: {csv_path}")
print(f"保存目录: {output_dir}")

# 读取CSV
df = pd.read_csv(csv_path)
print(f"成功加载CSV，共 {len(df)} 行数据")
print(f"列名: {list(df.columns)}")

# 收集所有唯一URL和相关的QA信息
all_urls = set()
url_to_qa = {}

for idx, row in df.iterrows():
    urls_str = row.get('golden_chunks_urls', '[]')
    if pd.notna(urls_str):
        try:
            urls = eval(urls_str) if isinstance(urls_str, str) else urls_str
            if isinstance(urls, list):
                for url in urls:
                    if isinstance(url, str) and url.startswith('http'):
                        all_urls.add(url)
                        if url not in url_to_qa:
                            url_to_qa[url] = []
                        url_to_qa[url].append({
                            'question': row.get('question', ''),
                            'answer': row.get('answer', ''),
                            'question_date': row.get('question_date', '')
                        })
        except Exception as e:
            continue

print(f"共 {len(all_urls)} 个唯一URL")

# 加载已爬取记录
crawled_urls = set()
record_file = output_dir / "crawled_urls.txt"
if record_file.exists():
    with open(record_file, 'r', encoding='utf-8') as f:
        crawled_urls = set(line.strip() for line in f)
    print(f"已有 {len(crawled_urls)} 个URL被爬取过")


def get_url_hash(url):
    """生成URL的哈希作为文件名"""
    return hashlib.md5(url.encode()).hexdigest()


def extract_publish_date(url, soup):
    """从URL或页面中提取发布日期"""
    # 尝试从URL中提取日期
    date_pattern = r'/(\d{4}-\d{2}-\d{2})/'
    match = re.search(date_pattern, url)
    if match:
        return match.group(1)

    # 尝试从页面中查找发布时间
    time_selectors = ['time', '.date', '.publish-time', '.article-time',
                      '[property="article:published_time"]', '.time']
    for selector in time_selectors:
        elem = soup.select_one(selector)
        if elem:
            date_text = elem.get('content', elem.text)
            # 提取日期格式
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', date_text)
            if date_match:
                return date_match.group(1)
    return None


def fetch_and_save_news(url):
    """爬取并保存新闻内容"""
    if url in crawled_urls:
        print(f"  跳过已爬取: {url[:60]}...")
        return None

    url_hash = get_url_hash(url)
    text_file = articles_dir / f"{url_hash}.txt"

    # 如果文件已存在，直接返回
    if text_file.exists():
        print(f"  文件已存在: {text_file.name}")
        return load_saved_article(url_hash)

    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        response.encoding = response.apparent_encoding

        html_content = response.text

        # 使用 trafilatura 提取正文（更干净）
        text = trafilatura.extract(html_content, include_comments=False, include_tables=True)

        if not text:
            # 备用：使用 BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            text = '\n'.join(line for line in lines if line)

        # 提取标题
        soup = BeautifulSoup(html_content, 'html.parser')
        title = soup.find('title')
        title = title.text.strip() if title else url.split('/')[-1]
        # 清理标题
        title = title.split('_')[0].split('|')[0].strip()

        # 提取发布时间
        publish_date = extract_publish_date(url, soup)

        # 获取相关的QA信息
        related_qa = url_to_qa.get(url, [])
        question_date = related_qa[0].get('question_date', '') if related_qa else ''

        # 保存为干净的文本文件
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(f"标题：{title}\n")
            f.write(f"来源：{url}\n")
            if publish_date:
                f.write(f"发布时间：{publish_date}\n")
            if question_date:
                f.write(f"相关问答时间：{question_date}\n")
            f.write(f"{'=' * 50}\n\n")
            f.write(text)

        # 保存元数据
        metadata = {
            'id': url_hash,
            'title': title,
            'url': url,
            'publish_date': publish_date,
            'content_length': len(text),
            'related_questions': len(related_qa)
        }

        # 记录已爬取
        with open(record_file, 'a', encoding='utf-8') as f:
            f.write(url + '\n')
        crawled_urls.add(url)

        print(f"  ✓ 成功: {title[:40]}... ({len(text)}字符)")

        return metadata

    except Exception as e:
        print(f"  ✗ 失败: {e}")
        return None


def load_saved_article(url_hash):
    """加载已保存的文章"""
    text_file = articles_dir / f"{url_hash}.txt"
    if text_file.exists():
        with open(text_file, 'r', encoding='utf-8') as f:
            content = f.read()
        lines = content.split('\n')
        title = lines[0].replace('标题：', '') if lines else ''
        return {
            'id': url_hash,
            'title': title,
            'file': str(text_file)
        }
    return None


# 爬取全部数据并保存
url_list = list(all_urls)
print(f"\n开始爬取并保存...\n")

results = []
for i, url in enumerate(url_list):
    print(f"{i + 1}. {url[:70]}...")
    result = fetch_and_save_news(url)
    if result:
        results.append(result)
    time.sleep(1)  # 避免请求太快

# 保存索引文件
index_file = output_dir / "index.json"
with open(index_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# 生成汇总报告
summary = {
    'total_urls': len(all_urls),
    'crawled_count': len(results),
    'save_directory': str(articles_dir),
    'articles': results
}

summary_file = output_dir / "summary.json"
with open(summary_file, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n{'=' * 50}")
print(f"爬取完成！")
print(f"成功: {len(results)}/10")
print(f"文章保存目录: {articles_dir}")
print(f"索引文件: {index_file}")
print(f"汇总报告: {summary_file}")

# 显示保存的文件列表
print(f"\n保存的文件:")
for r in results:
    print(f"  - {r['title'][:40]}... ({r['content_length']}字符)")