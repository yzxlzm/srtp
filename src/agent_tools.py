# 简单示例工具集合。真实项目中可接入浏览器、搜索、数据库等。


def web_search_sim(query: str) -> str:
    # 占位：在需要时可以替换为真实搜索 API
    return f"[模拟搜索] 关于 {query} 的摘要（请接入真实搜索接口以获得网络结果）。"


TOOLS = [
    {
        "name": "web_search",
        "func": web_search_sim,
        "description": "模拟网络搜索并返回摘要",
    }
]

