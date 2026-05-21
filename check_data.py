import os
import json
import pandas as pd
def explore_dataset(dataset_path):
    """探索ChronoQA数据集的结构和内容"""

    print("=" * 60)
    print(f"数据集路径: {dataset_path}")
    print("=" * 60)

    # 检查路径是否存在
    if not os.path.exists(dataset_path):
        print(f"错误：路径不存在 - {dataset_path}")
        return

    # 获取目录结构
    print("\n1. 目录结构:")
    print("-" * 40)
    for root, dirs, files in os.walk(dataset_path):
        level = root.replace(dataset_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        # 只显示前5个文件作为示例
        for i, file in enumerate(files[:5]):
            file_size = os.path.getsize(os.path.join(root, file))
            size_str = f"({file_size} bytes)" if file_size < 1024 else f"({file_size / 1024:.1f} KB)"
            print(f"{subindent}{file} {size_str}")
        if len(files) > 5:
            print(f"{subindent}... 还有 {len(files) - 5} 个文件")

    # 统计文件类型
    print("\n2. 文件类型统计:")
    print("-" * 40)
    file_types = {}
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext:
                file_types[ext] = file_types.get(ext, 0) + 1
            else:
                file_types['no_extension'] = file_types.get('no_extension', 0) + 1

    for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {ext}: {count} 个文件")

    # 查找并显示JSON/CSV文件示例
    print("\n3. 数据文件内容预览:")
    print("-" * 40)

    # 查找第一个JSON文件
    json_files = []
    csv_files = []
    txt_files = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
            elif file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
            elif file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))

    # 预览JSON文件
    if json_files:
        print(f"\n找到 {len(json_files)} 个JSON文件，预览第一个:")
        try:
            with open(json_files[0], 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    print(f"  文件: {os.path.basename(json_files[0])}")
                    print(f"  类型: 列表，包含 {len(data)} 个元素")
                    if len(data) > 0:
                        print(f"  第一个元素示例:")
                        print(f"  {json.dumps(data[0], ensure_ascii=False, indent=4)[:500]}")
                elif isinstance(data, dict):
                    print(f"  文件: {os.path.basename(json_files[0])}")
                    print(f"  类型: 字典，包含 {len(data)} 个键")
                    print(f"  键列表: {list(data.keys())[:10]}")
                    print(f"  内容示例:")
                    print(f"  {json.dumps(data, ensure_ascii=False, indent=4)[:500]}")
        except Exception as e:
            print(f"  读取JSON文件出错: {e}")

    # 预览CSV文件
    if csv_files:
        print(f"\n找到 {len(csv_files)} 个CSV文件，预览第一个:")
        try:
            df = pd.read_csv(csv_files[0], nrows=5)
            print(f"  文件: {os.path.basename(csv_files[0])}")
            print(f"  形状: {df.shape}")
            print(f"  列名: {list(df.columns)}")
            print(f"  前几行数据:")
            print(df.head())
        except Exception as e:
            print(f"  读取CSV文件出错: {e}")

    # 预览TXT文件
    if txt_files:
        print(f"\n找到 {len(txt_files)} 个TXT文件，预览第一个:")
        try:
            with open(txt_files[0], 'r', encoding='utf-8') as f:
                content = f.read(500)
                print(f"  文件: {os.path.basename(txt_files[0])}")
                print(f"  前500字符:")
                print(content)
        except Exception as e:
            print(f"  读取TXT文件出错: {e}")

    # 总体统计
    print("\n4. 总体统计:")
    print("-" * 40)
    total_files = 0
    total_size = 0
    for root, dirs, files in os.walk(dataset_path):
        total_files += len(files)
        for file in files:
            total_size += os.path.getsize(os.path.join(root, file))

    print(f"  总文件数: {total_files}")
    print(f"  总大小: {total_size / 1024 / 1024:.2f} MB")

    # 显示README文件内容（如果有）
    readme_files = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if 'readme' in file.lower():
                readme_files.append(os.path.join(root, file))

    if readme_files:
        print("\n5. README文件内容（前500字符）:")
        print("-" * 40)
        try:
            with open(readme_files[0], 'r', encoding='utf-8') as f:
                content = f.read(500)
                print(content)
        except Exception as e:
            print(f"  读取README文件出错: {e}")


if __name__ == "__main__":
    # 使用你提供的数据集地址
    dataset_path = r"F:\下载\ChronoQA-main"
    explore_dataset(dataset_path)