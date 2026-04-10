from fastapi import Header,Depends,HTTPException,status
from sqlalchemy.ext.asyncio import AsyncSession
from config.DB_config import get_db
from crud.users import get_user_by_token 






async def get_current_user(db:AsyncSession=Depends(get_db),authorization:str=Header(...,alias='Authorization')):
    '''
    根据token查询用户，返回用户
    '''
    # 两种从前端获取token的方法
    token = authorization
    
    # token = authorization.replace("Bearer ", "")
    
    user = await get_user_by_token(db,token)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,detail='无效的令牌或已过期的令牌')
    
    return user