#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import glob

def fix_liquid_syntax(file_path):
    """Liquid 문법 오류를 수정합니다."""
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
        
        # {{ }} 패턴을 {% raw %}{{ }}{% endraw %}로 감싸기
        # 단, 이미 {% raw %}로 감싸진 것은 제외
        content = re.sub(
            r'(?<!{% raw %})\{\{([^}]+)\}\}(?!{% endraw %})',
            r'{% raw %}{{\1}}{% endraw %}',
            content
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
    print("Liquid 문법 오류 수정 시작...")
    
    # 문제가 있는 파일들
    problem_files = [
        "2025-01-18-project-django-balance-game.md",
        "2025-01-18-project-django-instagram-clone.md", 
        "2025-01-18-web-django-1n-relation.md",
        "2025-01-18-web-django-auth.md",
        "2025-01-18-web-django-basics.md",
        "2025-01-18-web-django-comment.md",
        "2025-01-18-web-django-crud-update.md",
        "2025-01-18-web-django-crud.md",
        "2025-01-18-web-django-image-upload.md",
        "2025-01-18-web-django-mn-relation.md",
        "2025-01-18-web-django-modelform.md",
        "2025-01-18-web-javascript-basics.md"
    ]
    
    success_count = 0
    total_count = len(problem_files)
    
    for filename in problem_files:
        file_path = f"_posts/{filename}"
        if os.path.exists(file_path):
            if fix_liquid_syntax(file_path):
                success_count += 1
        else:
            print(f"FILE NOT FOUND: {filename}")
    
    print(f"\n수정 완료: {success_count}/{total_count} 파일")

if __name__ == "__main__":
    main()
