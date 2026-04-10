"""
启动 Web 版本的问答页面。
运行：python src/main.py，然后浏览器访问 http://127.0.0.1:8000
"""

import uvicorn


def main():
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()

