"""
构建机器学习文档全量知识图谱
运行：python src/build_ml_kg.py
"""

from src.knowledge_graph import build_ml_full_kg


def main():
    """构建机器学习文档全量知识图谱"""
    print("="*60)
    print("机器学习知识图谱构建工具")
    print("="*60)
    print()
    
    try:
        # 构建全量知识图谱
        kg = build_ml_full_kg(
            docx_filename="【电子书】周志华-机器学习.docx",
            force_rebuild=False  # 设置为True可强制重建
        )
        
        print("\n✓ 知识图谱构建完成！")
        print("\n提示：")
        print("  - 知识图谱已保存到缓存目录")
        print("  - 可以使用 query_knowledge_graph 函数查询知识图谱")
        print("  - 可以使用 generate_answer_from_kg 函数基于知识图谱生成回答")
        print("  - 运行 python src/generate_from_kg_example.py 查看使用示例")
        print("  - 如需重新构建，请设置 force_rebuild=True")
        
    except Exception as e:
        print(f"\n✗ 构建知识图谱时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

