import json
import re
from pathlib import Path

# 广告关键词
AD_MARKERS = [
    'VIP课程推荐', 'APP专享直播', '热门推荐', '责任编辑',
    '举报邮箱', 'Copyright', '违法和不良信息举报',
    '扫一扫', '欢迎发表评论', '新浪科技公众号',
    '新浪财经声明', '不构成投资建议', '仅供参考'
]

# 无效页面关键词
INVALID_KEYWORDS = [
    '页面没有找到', '404', '页面不存在',
    '5秒钟之后将会带您进入新浪首页', 'Not Found'
]


def is_valid_url(url: str) -> bool:
    """判断URL是否有效（以http或https开头）"""
    if not url:
        return False
    url = url.strip()
    return url.startswith('http://') or url.startswith('https://')


def parse_txt_file(content: str) -> dict:
    """解析TXT文件，提取元数据和正文"""
    lines = content.split('\n')

    result = {
        'title': '',
        'url': '',
        'publish_date': '',
        'qa_date': '',
        'content': ''
    }

    content_start = False
    content_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 解析标题行
        if line.startswith('标题：'):
            result['title'] = line.replace('标题：', '').strip()
        # 解析来源行
        elif line.startswith('来源：'):
            result['url'] = line.replace('来源：', '').strip()
        # 解析发布时间
        elif line.startswith('发布时间：'):
            result['publish_date'] = line.replace('发布时间：', '').strip()
        # 解析问答时间
        elif line.startswith('相关问答时间：'):
            result['qa_date'] = line.replace('相关问答时间：', '').strip()
        # 分割线表示正文开始
        elif '===' in line or '---' in line:
            content_start = True
        # 正文内容
        elif content_start:
            # 移除原标题行（如果出现在正文中）
            if not line.startswith('原标题：'):
                content_lines.append(line)

    result['content'] = '\n'.join(content_lines)
    return result


def clean_content(content: str) -> str:
    """清洗正文内容"""
    if not content:
        return ""

    # 删除广告行
    lines = content.split('\n')
    cleaned_lines = []

    for line in lines:
        is_ad = False
        for marker in AD_MARKERS:
            if marker in line:
                is_ad = True
                break
        if not is_ad and len(line.strip()) > 10:
            cleaned_lines.append(line.strip())

    return '\n'.join(cleaned_lines)


def process_all(input_dir: str, output_dir: str, invalid_dir: str = None):
    """处理所有TXT文件，只保留有效URL的新闻"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    if invalid_dir:
        invalid_path = Path(invalid_dir)
        invalid_path.mkdir(exist_ok=True)
    else:
        invalid_path = output_path / "invalid"
        invalid_path.mkdir(exist_ok=True)

    # 获取所有TXT文件
    files = list(input_path.glob("*.txt"))
    print(f"找到 {len(files)} 个TXT文件")

    valid_count = 0
    invalid_count = 0
    url_invalid_count = 0  # 新增：URL无效计数

    for i, file_path in enumerate(files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()
        except Exception as e:
            print(f"读取失败: {file_path.name}, {e}")
            invalid_count += 1
            continue

        # 检查是否无效页面
        is_invalid = any(kw in raw_content for kw in INVALID_KEYWORDS)
        if is_invalid:
            dest = invalid_path / file_path.name
            file_path.rename(dest)
            invalid_count += 1
            continue

        # 解析文件
        parsed = parse_txt_file(raw_content)

        # 检查URL是否有效（新增过滤条件）
        if not is_valid_url(parsed['url']):
            dest = invalid_path / f"invalid_url_{file_path.name}"
            file_path.rename(dest)
            url_invalid_count += 1
            continue

        # 检查是否有有效内容
        if len(parsed['content']) < 50:
            dest = invalid_path / f"too_short_{file_path.name}"
            file_path.rename(dest)
            invalid_count += 1
            continue

        # 清洗正文
        cleaned_content = clean_content(parsed['content'])

        if len(cleaned_content) < 30:
            dest = invalid_path / f"empty_{file_path.name}"
            file_path.rename(dest)
            invalid_count += 1
            continue

        # 保存清洗后的文件（JSON格式）
        output_data = {
            'title': parsed['title'],
            'url': parsed['url'],
            'publish_date': parsed['publish_date'],
            'qa_date': parsed['qa_date'],
            'content': cleaned_content
        }

        output_file = output_path / f"{file_path.stem}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        valid_count += 1

        if (i + 1) % 200 == 0:
            print(f"进度: {i + 1}/{len(files)} | 有效: {valid_count} | URL无效: {url_invalid_count} | 其他无效: {invalid_count}")

    print(f"\n{'=' * 50}")
    print(f"处理完成！")
    print(f"有效文件（有有效URL）: {valid_count}")
    print(f"URL无效文件（已过滤）: {url_invalid_count}")
    print(f"其他无效文件: {invalid_count}")
    print(f"输出目录: {output_path}")
    print(f"输出格式: JSON")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    process_all(
        input_dir=r"F:\Pycharm\RAG\news_corpus_simple\articles",  # TXT文件目录
        output_dir=r"F:\Pycharm\RAG\news_corpus_simple\cleaned",  # 清洗后JSON目录
        invalid_dir=r"F:\Pycharm\RAG\news_corpus_simple\invalid"  # 无效页面目录
    )