import os
import re
import glob
import yaml

def fix_post_file(file_path):
    """
    주어진 Markdown 파일의 front matter를 Chirpy 형식으로 수정합니다.
    - 'excerpt' 필드 제거
    - 'last_modified_at' 필드 제거
    - 'permalink' 필드 제거
    - 'toc_sticky' 필드 제거
    - 'date' 필드에 '+0900' 타임존 정보 추가 (없을 경우)
    - 'comments: false', 'mermaid: true', 'math: true' 필드 추가 (없을 경우)
    - 'toc: true' 필드 추가 (없을 경우)
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

        # 제거할 필드들
        fields_to_remove = ['excerpt', 'last_modified_at', 'permalink', 'toc_sticky']
        for field in fields_to_remove:
            if field in front_matter:
                del front_matter[field]

        # 'date' 필드에 '+0900' 타임존 정보 추가
        if 'date' in front_matter and isinstance(front_matter['date'], (str, type(None))):
            date_str = str(front_matter['date']) if front_matter['date'] else ''
            if date_str and not re.search(r'\+\d{4}$', date_str):
                front_matter['date'] = f"{date_str} +0900"
        elif 'date' not in front_matter:
            # 날짜 필드가 없는 경우 기본값 추가
            front_matter['date'] = "2025-01-01 00:00:00 +0900"

        # 필수 필드 추가
        if 'comments' not in front_matter:
            front_matter['comments'] = False
        if 'mermaid' not in front_matter:
            front_matter['mermaid'] = True
        if 'math' not in front_matter:
            front_matter['math'] = True
        if 'toc' not in front_matter:
            front_matter['toc'] = True

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
    print("새로 추가된 파일들의 Front matter 수정 시작...")

    # 새로 추가된 파일들만 처리
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
            if fix_post_file(file_path):
                success_count += 1
        else:
            print(f"NOT FOUND: {filename}")

    print(f"\nFront matter 수정 완료: {success_count}/{total_count} 파일")

if __name__ == "__main__":
    main()
