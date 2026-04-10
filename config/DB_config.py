from sqlalchemy.ext.asyncio import create_async_engine,async_sessionmaker,AsyncSession


# 创建数据库URL
ASYNC_DATABASE_URL ="mysql+aiomysql://root:Wgh-103009@localhost:3306/news_app?charset=utf8mb4"


# 创建异步引擎：
async_engine = create_async_engine(
    ASYNC_DATABASE_URL,
    echo = True,
    pool_size = 10,
    max_overflow = 20
)


# 异步会话 创建
AsyncSessionLocal = async_sessionmaker(
    bind = async_engine,
    class_ = AsyncSession,
    expire_on_commit = False
)

# 创建依赖项 获取数据库会话
async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()