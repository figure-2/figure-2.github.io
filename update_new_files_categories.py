import os
import re
import yaml

def update_post_categories(file_path):
    """
    주어진 Markdown 파일의 front matter에서 카테고리를 업데이트합니다.
    """
    content = None
    encodings = ['utf-8', 'cp949', 'euc-kr', 'latin-1']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            break
        except UnicodeDecodeError:
            continue

    if content is None:
        print(f"ENCODING ERROR: {os.path.basename(file_path)}")
        return False

    front_matter_match = re.match(r'---\s*\n(.*?)\n---\s*\n(.*)', content, re.DOTALL)
    if not front_matter_match:
        return False

    front_matter_str = front_matter_match.group(1)
    body = front_matter_match.group(2)

    try:
        front_matter = yaml.safe_load(front_matter_str)
        if front_matter is None:
            front_matter = {}

        # 카테고리 매핑
        category_mapping = {
            "PYTHON": "02_PYTHON",
            "PANDAS": "03_PANDAS", 
            "WEB": "04_WEB",
            "WEB_DEVELOPMENT": "04_WEB_DEVELOPMENT",
            "SPECIAL": "14_SPECIAL",
            "02.Python": "02_PYTHON",
            "02_Python": "02_PYTHON"
        }

        if 'categories' in front_matter and isinstance(front_matter['categories'], list):
            updated_categories = []
            for category in front_matter['categories']:
                # 매핑에 있는 카테고리만 업데이트
                if category in category_mapping:
                    updated_categories.append(category_mapping[category])
                else:
                    updated_categories.append(category)
            front_matter['categories'] = updated_categories

        updated_front_matter_str = yaml.dump(front_matter, allow_unicode=True, default_flow_style=False, sort_keys=False)
        content = f"---\n{updated_front_matter_str.strip()}\n---\n{body}"

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"OK: {os.path.basename(file_path)}")
        return True

    except Exception as e:
        print(f"ERROR: {os.path.basename(file_path)} - {str(e)}")
        return False

def main():
    print("새로 추가된 파일들의 카테고리 업데이트 시작...")
    
    # 새로 추가된 파일들
    new_files = [
        "2025-01-18-pandas-missing-values.md",
        "2025-01-18-pandas-data-structures.md",
        "2025-01-18-pandas-preprocessing.md",
        "2025-01-18-pandas-query-sort-filter.md",
        "2025-01-18-pandas-statistics.md",
        "2025-01-18-pandas-file-io.md",
        "2025-01-18-python-functions.md",
        "2025-01-18-web-html-fundamentals.md",
        "2025-01-19-python-function-arguments.md",
        "2025-01-21-python-recursion.md",
        "2025-09-20-git-advanced.md",
        "2025-09-20-git-basics.md",
        "2025-09-20-python-basics.md",
        "2025-09-20-virtual-environment-setup.md",
        "2025-10-02-pj-parer-comparison.md"
    ]

    success_count = 0
    total_count = len(new_files)

    for filename in new_files:
        file_path = f"_posts/{filename}"
        if os.path.exists(file_path):
            if update_post_categories(file_path):
                success_count += 1
        else:
            print(f"NOT FOUND: {filename}")

    print(f"\n카테고리 업데이트 완료: {success_count}/{total_count} 파일")

if __name__ == "__main__":
    main()
