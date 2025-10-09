import os
import shutil

def copy_missing_files():
    """누락된 파일들을 backup에서 현재 디렉토리로 복사합니다."""
    
    # 누락된 파일 목록
    missing_files = [
        "2025-01-18-pandas-결측치.md",
        "2025-01-18-pandas-자료구조.md", 
        "2025-01-18-pandas-전처리.md",
        "2025-01-18-pandas-조회정렬조건필터.md",
        "2025-01-18-pandas-통계.md",
        "2025-01-18-pandas-파일입출력.md",
        "2025-01-18-python-함수.md",
        "2025-01-18-web-html-기초.md",
        "2025-01-19-python-함수의인수.md",
        "2025-01-20-python-lambda-타입힌트-이름공간.md",
        "2025-01-21-python-재귀.md",
        "2025-09-20-git-고급.md",
        "2025-09-20-git-기초.md",
        "2025-09-20-python-기초.md",
        "2025-09-20-가상환경-설정.md",
        "2025-10-02-PJ_Parer 비교.md"
    ]
    
    source_dir = "../figure-2.github.io-backup/_posts"
    target_dir = "_posts"
    
    success_count = 0
    
    for filename in missing_files:
        source_path = os.path.join(source_dir, filename)
        target_path = os.path.join(target_dir, filename)
        
        try:
            if os.path.exists(source_path):
                shutil.copy2(source_path, target_path)
                print(f"OK: {filename}")
                success_count += 1
            else:
                print(f"NOT FOUND: {filename}")
        except Exception as e:
            print(f"ERROR: {filename} - {str(e)}")
    
    print(f"\n복사 완료: {success_count}/{len(missing_files)} 파일")

if __name__ == "__main__":
    copy_missing_files()
