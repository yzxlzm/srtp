import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from model_class.users import Base

# 数据库连接URL
ASYNC_DATABASE_URL = "mysql+aiomysql://root:Wgh-103009@localhost:3306/news_app?charset=utf8mb4"

async def create_tables():
    engine = create_async_engine(ASYNC_DATABASE_URL, echo=True)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    await engine.dispose()
    print("✓ 数据库表创建成功！")

if __name__ == "__main__":
    asyncio.run(create_tables())
