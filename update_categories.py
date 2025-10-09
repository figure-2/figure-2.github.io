#!/usr/bin/env python3
"""
포스트 파일의 카테고리를 새로운 형식으로 업데이트하는 스크립트
"""

import os
import re
import glob
import yaml

# 카테고리 매핑 정의
CATEGORY_MAPPING = {
    "WEB_DEVELOPMENT": "04_WEB_DEVELOPMENT",
    "WEB": "04_WEB",
    "PYTHON": "02_PYTHON",
    "SQL": "05_SQL",
    "SPECIAL": "14_SPECIAL",
    "DATA_ANALYSIS": "07_DATA_ANALYSIS",
    "Pandas": "03_PANDAS",
    "MACHINE_LEARNING": "08_MACHINE_LEARNING",
    "CRUD": "06_CRUD",
    "M:N관계": "06_MN_RELATION",
    "OOP": "02_OOP",
}

def update_post_file(file_path):
    """
    주어진 Markdown 파일의 카테고리를 업데이트합니다.
    """
    content = None
    encodings = ['utf-8', 'cp949', 'euc-kr', 'latin-1']

    # 파일 읽기 (여러 인코딩 시도)
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

    # front matter와 본문 분리
    front_matter_match = re.match(r'---\s*\n(.*?)\n---\s*\n(.*)', content, re.DOTALL)
    if not front_matter_match:
        print(f"FRONT MATTER ERROR: {os.path.basename(file_path)} - Front matter not found or malformed.")
        return False

    front_matter_str = front_matter_match.group(1)
    body = front_matter_match.group(2)

    try:
        front_matter = yaml.safe_load(front_matter_str)
        if front_matter is None:
            front_matter = {}

        # 카테고리 업데이트
        if 'categories' in front_matter:
            updated_categories = []
            for category in front_matter['categories']:
                new_category = CATEGORY_MAPPING.get(category, category)
                updated_categories.append(new_category)
            front_matter['categories'] = updated_categories

        # 수정된 front matter를 YAML 문자열로 변환
        updated_front_matter_str = yaml.dump(front_matter, allow_unicode=True, default_flow_style=False, sort_keys=False)

        # 새로운 내용 조합
        content = f"---\n{updated_front_matter_str.strip()}\n---\n{body}"

        # 파일 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"OK: {os.path.basename(file_path)}")
        return True

    except Exception as e:
        print(f"ERROR: {os.path.basename(file_path)} - {str(e)}")
        return False

def main():
    """메인 함수"""
    print("포스트 파일 카테고리 업데이트 시작...")

    # _posts 폴더의 모든 .md 파일 처리
    post_files = glob.glob('_posts/*.md')

    success_count = 0
    total_count = len(post_files)

    for file_path in post_files:
        if update_post_file(file_path):
            success_count += 1

    print(f"\n업데이트 완료: {success_count}/{total_count} 파일")

if __name__ == "__main__":
    main()
