import os
import shutil

def rename_korean_files():
    """한글 파일명을 영어로 변환합니다."""
    
    # 파일명 매핑 (한글 -> 영어)
    file_mapping = {
        "2025-01-18-pandas-결측치.md": "2025-01-18-pandas-missing-values.md",
        "2025-01-18-pandas-자료구조.md": "2025-01-18-pandas-data-structures.md",
        "2025-01-18-pandas-전처리.md": "2025-01-18-pandas-preprocessing.md",
        "2025-01-18-pandas-조회정렬조건필터.md": "2025-01-18-pandas-query-sort-filter.md",
        "2025-01-18-pandas-통계.md": "2025-01-18-pandas-statistics.md",
        "2025-01-18-pandas-파일입출력.md": "2025-01-18-pandas-file-io.md",
        "2025-01-18-python-함수.md": "2025-01-18-python-functions.md",
        "2025-01-18-web-html-기초.md": "2025-01-18-web-html-fundamentals.md",
        "2025-01-19-python-함수의인수.md": "2025-01-19-python-function-arguments.md",
        "2025-01-20-python-lambda-타입힌트-이름공간.md": "2025-01-20-python-lambda-type-hint-namespace.md",
        "2025-01-21-python-재귀.md": "2025-01-21-python-recursion.md",
        "2025-09-20-git-고급.md": "2025-09-20-git-advanced.md",
        "2025-09-20-git-기초.md": "2025-09-20-git-basics.md",
        "2025-09-20-python-기초.md": "2025-09-20-python-basics.md",
        "2025-09-20-가상환경-설정.md": "2025-09-20-virtual-environment-setup.md",
        "2025-10-02-PJ_Parer 비교.md": "2025-10-02-pj-parer-comparison.md"
    }
    
    posts_dir = "_posts"
    success_count = 0
    
    for old_name, new_name in file_mapping.items():
        old_path = os.path.join(posts_dir, old_name)
        new_path = os.path.join(posts_dir, new_name)
        
        try:
            if os.path.exists(old_path):
                os.rename(old_path, new_path)
                print(f"OK: {old_name} -> {new_name}")
                success_count += 1
            else:
                print(f"NOT FOUND: {old_name}")
        except Exception as e:
            print(f"ERROR: {old_name} - {str(e)}")
    
    print(f"\n파일명 변환 완료: {success_count}/{len(file_mapping)} 파일")

if __name__ == "__main__":
    rename_korean_files()
