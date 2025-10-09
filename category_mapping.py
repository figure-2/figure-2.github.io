#!/usr/bin/env python3
"""
카테고리 매핑 스크립트
기존 카테고리를 새로운 숫자 접두사 카테고리로 변환
"""

# 카테고리 매핑 정의
CATEGORY_MAPPING = {
    # 기존 카테고리 → 새로운 카테고리
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

def get_new_category(old_category):
    """기존 카테고리를 새로운 카테고리로 변환"""
    return CATEGORY_MAPPING.get(old_category, old_category)

def print_mapping():
    """매핑 결과 출력"""
    print("=== 카테고리 매핑 결과 ===")
    for old, new in CATEGORY_MAPPING.items():
        print(f"{old:20} → {new}")
    
    print(f"\n총 {len(CATEGORY_MAPPING)}개 카테고리 매핑 완료")

if __name__ == "__main__":
    print_mapping()
