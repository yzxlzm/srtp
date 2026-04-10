"""
知识图谱查询示例
展示如何使用各种查询方法查询知识图谱

运行：python src/query_kg_example.py
"""

import os
import tempfile
from pathlib import Path
from knowledge_graph import (
    KnowledgeGraph,
    build_ml_full_kg,
    query_knowledge_graph,
    query_entity_by_name,
    query_relations_by_entity,
    query_entities_by_type,
    query_relations_by_type,
    get_entity_neighbors,
    format_query_results
)


def load_ml_kg():
    """加载机器学习知识图谱"""
    base_dir = Path(__file__).resolve().parent.parent
    
    # 缓存目录
    env_cache = os.environ.get("VECTOR_CACHE_DIR")
    if env_cache:
        cache_dir = Path(env_cache).parent / "kg_cache"
    else:
        cache_dir = Path(tempfile.gettempdir()) / "kg_cache"
    
    kg_path = cache_dir / "ml_full_kg"
    
    kg = KnowledgeGraph()
    try:
        kg.load(kg_path)
        print(f"✓ 成功加载知识图谱：{kg_path}")
        return kg
    except Exception as e:
        print(f"✗ 加载知识图谱失败: {e}")
        print("提示：请先运行 python src/build_ml_kg.py 构建知识图谱（全量）")
        return None


def example_1_keyword_query(kg: KnowledgeGraph):
    """示例1：关键词查询"""
    print("\n" + "="*60)
    print("示例1：关键词查询")
    print("="*60)
    
    query = "监督学习"
    print(f"\n查询关键词：{query}\n")
    
    results = query_knowledge_graph(kg, query)
    print(format_query_results(results, max_results=10))


def example_2_entity_query(kg: KnowledgeGraph):
    """示例2：精确查询实体"""
    print("\n" + "="*60)
    print("示例2：精确查询实体")
    print("="*60)
    
    # 先通过关键词查询找到一些实体
    results = query_knowledge_graph(kg, "学习")
    if results:
        # 取第一个实体
        entity_name = results[0].get("name", "")
        if entity_name:
            print(f"\n查询实体：{entity_name}\n")
            
            entity = query_entity_by_name(kg, entity_name)
            if entity:
                print(f"实体名称：{entity['name']}")
                print(f"实体类型：{entity['data'].get('type', '未知')}")
                print(f"实体属性：{entity['data'].get('properties', {})}")
            else:
                print("未找到该实体")


def example_3_entity_relations(kg: KnowledgeGraph):
    """示例3：查询实体的所有关系"""
    print("\n" + "="*60)
    print("示例3：查询实体的所有关系")
    print("="*60)
    
    # 先通过关键词查询找到一些实体
    results = query_knowledge_graph(kg, "算法")
    if results:
        entity_name = results[0].get("name", "")
        if entity_name:
            print(f"\n查询实体：{entity_name} 的所有关系\n")
            
            relations = query_relations_by_entity(kg, entity_name)
            print(f"找到 {len(relations)} 个关系：\n")
            
            for i, rel in enumerate(relations[:10], 1):  # 只显示前10个
                print(f"  {i}. {rel['head']} --[{rel['relation']}]--> {rel['tail']}")


def example_4_entity_type_query(kg: KnowledgeGraph):
    """示例4：按实体类型查询"""
    print("\n" + "="*60)
    print("示例4：按实体类型查询")
    print("="*60)
    
    # 先获取统计信息，看看有哪些实体类型
    stats = kg.get_statistics()
    entity_types = list(stats.get("entity_types", {}).keys())
    
    if entity_types:
        entity_type = entity_types[0]  # 取第一个类型
        print(f"\n查询实体类型：{entity_type}\n")
        
        entities = query_entities_by_type(kg, entity_type)
        print(f"找到 {len(entities)} 个 {entity_type} 类型的实体：\n")
        
        for i, entity in enumerate(entities[:10], 1):  # 只显示前10个
            print(f"  {i}. {entity['name']}")


def example_5_relation_type_query(kg: KnowledgeGraph):
    """示例5：按关系类型查询"""
    print("\n" + "="*60)
    print("示例5：按关系类型查询")
    print("="*60)
    
    # 先获取统计信息，看看有哪些关系类型
    stats = kg.get_statistics()
    relation_types = list(stats.get("relation_types", {}).keys())
    
    if relation_types:
        relation_type = relation_types[0]  # 取第一个类型
        print(f"\n查询关系类型：{relation_type}\n")
        
        relations = query_relations_by_type(kg, relation_type)
        print(f"找到 {len(relations)} 个 {relation_type} 类型的关系：\n")
        
        for i, rel in enumerate(relations[:10], 1):  # 只显示前10个
            print(f"  {i}. {rel['head']} --[{rel['relation']}]--> {rel['tail']}")


def example_6_entity_neighbors(kg: KnowledgeGraph):
    """示例6：查询实体的邻居节点"""
    print("\n" + "="*60)
    print("示例6：查询实体的邻居节点")
    print("="*60)
    
    # 先通过关键词查询找到一些实体
    results = query_knowledge_graph(kg, "学习")
    if results:
        entity_name = results[0].get("name", "")
        if entity_name:
            print(f"\n查询实体：{entity_name} 的邻居节点\n")
            
            neighbors = get_entity_neighbors(kg, entity_name)
            
            print(f"实体：{neighbors['entity']}")
            print(f"\n出边关系（{len(neighbors['outgoing'])} 个）：")
            for i, rel in enumerate(neighbors['outgoing'][:5], 1):
                print(f"  {i}. {rel['head']} --[{rel['relation']}]--> {rel['tail']}")
            
            print(f"\n入边关系（{len(neighbors['incoming'])} 个）：")
            for i, rel in enumerate(neighbors['incoming'][:5], 1):
                print(f"  {i}. {rel['head']} --[{rel['relation']}]--> {rel['tail']}")
            
            print(f"\n连接的实体（{len(neighbors['connected_entities'])} 个）：")
            for i, entity in enumerate(list(neighbors['connected_entities'])[:10], 1):
                print(f"  {i}. {entity}")


def example_7_statistics(kg: KnowledgeGraph):
    """示例7：查看知识图谱统计信息"""
    print("\n" + "="*60)
    print("示例7：知识图谱统计信息")
    print("="*60)
    
    stats = kg.get_statistics()
    
    print(f"\n总实体数：{stats['total_entities']}")
    print(f"总关系数：{stats['total_relations']}")
    
    print(f"\n实体类型分布：")
    for entity_type, count in sorted(stats['entity_types'].items(), 
                                     key=lambda x: x[1], reverse=True):
        print(f"  - {entity_type}: {count}")
    
    print(f"\n关系类型分布：")
    for relation_type, count in sorted(stats['relation_types'].items(), 
                                       key=lambda x: x[1], reverse=True):
        print(f"  - {relation_type}: {count}")


def interactive_query(kg: KnowledgeGraph):
    """交互式查询"""
    print("\n" + "="*60)
    print("交互式查询模式")
    print("="*60)
    print("\n输入查询关键词（输入 'quit' 退出）：")
    
    while True:
        try:
            query = input("\n> ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                print("退出交互式查询")
                break
            
            if not query:
                continue
            
            results = query_knowledge_graph(kg, query)
            print("\n" + format_query_results(results, max_results=15))
            
        except KeyboardInterrupt:
            print("\n\n退出交互式查询")
            break
        except Exception as e:
            print(f"\n查询出错: {e}")


def main():
    """主函数"""
    print("="*60)
    print("知识图谱查询示例")
    print("="*60)
    
    # 加载知识图谱
    kg = load_ml_kg()
    if kg is None:
        return
    
    # 显示统计信息
    example_7_statistics(kg)
    
    # 运行各种查询示例
    example_1_keyword_query(kg)
    example_2_entity_query(kg)
    example_3_entity_relations(kg)
    example_4_entity_type_query(kg)
    example_5_relation_type_query(kg)
    example_6_entity_neighbors(kg)
    
    # 交互式查询
    print("\n" + "="*60)
    choice = input("\n是否进入交互式查询模式？(y/n): ").strip().lower()
    if choice == 'y':
        interactive_query(kg)


if __name__ == "__main__":
    main()

