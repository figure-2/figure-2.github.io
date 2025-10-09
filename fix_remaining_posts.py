#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob

def fix_post_file(file_path):
    """포스트 파일을 수정합니다."""
    try:
        # 여러 인코딩으로 시도
        encodings = ['utf-8', 'cp949', 'euc-kr', 'latin-1']
        content = None
        
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
        
        # excerpt 필드 제거
        content = re.sub(r'^excerpt:.*\n', '', content, flags=re.MULTILINE)
        
        # last_modified_at 필드 제거
        content = re.sub(r'^last_modified_at:.*\n', '', content, flags=re.MULTILINE)
        
        # date 필드에 +0900 추가 (없는 경우에만)
        date_pattern = r'^date:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})(?!\s+\+\d{4})'
        content = re.sub(date_pattern, r'date: \1 +0900', content, flags=re.MULTILINE)
        
        # comments, mermaid, math 필드 추가 (없는 경우에만)
        if 'comments: false' not in content:
            # date 필드 다음에 추가
            content = re.sub(
                r'^(date:.*\+0900)$',
                r'\1\ntoc: true\ncomments: false\nmermaid: true\nmath: true',
                content,
                flags=re.MULTILINE
            )
        
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
    print("포스트 파일 수정 시작...")
    
    # _posts 폴더의 모든 .md 파일 처리
    post_files = glob.glob('_posts/*.md')
    
    success_count = 0
    total_count = len(post_files)
    
    for file_path in post_files:
        if fix_post_file(file_path):
            success_count += 1
    
    print(f"\n수정 완료: {success_count}/{total_count} 파일")

if __name__ == "__main__":
    main()
