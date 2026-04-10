from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid
from  datetime import datetime,timedelta
from fastapi import HTTPException
from model_class.users import User,UserToken
from schemes.users import UserRequest
from utils.security import get_hashed_password,verify_password



async def get_user_by_username(db:AsyncSession,username:str):
    query = select(User).where(User.username == username)
    res = await db.execute(query)
    
    return res.scalar_one_or_none()

async def create_user(db:AsyncSession,user_data:UserRequest):
    '''
    创建用户，并将密码加密
    '''
    # 密码加密：
    hashed_password = get_hashed_password(user_data.password)
    user = User(username=user_data.username,password = hashed_password)
    
    db.add(user)
    
    await db.commit()
    await db.refresh(user)
    return user

async def create_token(db:AsyncSession,user_id:int):
    '''
    生成用户令牌，
    设置token过期时间，查询token是否存在 有：更新 无：创建
    '''
    token = str(uuid.uuid4())
    
    expires_at = datetime.now() + timedelta(days=7)
    
    query = select(UserToken).where(UserToken.user_id ==user_id)
    res =await db.execute(query)
    user_token = res.scalar_one_or_none()
    
    if user_token :
        user_token.token = token
        user_token.expires_at = expires_at
    else:
        user_token = UserToken(token=token,user_id=user_id,expires_at=expires_at)
        db.add(user_token)

    await db.commit()
    await db.refresh(user_token)
    return token
        
async def authenticate_users(db:AsyncSession,username:str,password:str):
    query = select(User).where(User.username == username)
    res =await db.execute(query)
    user = res.scalar_one_or_none()
    
    if  not user:
            return None
    if not verify_password(password,user.password):
        return None
    
    return user # 验证成功
    
async def get_user_by_token(db:AsyncSession,token:str):
    '''
    通过token查询到user
    '''
    
    query = select(UserToken).where(UserToken.token ==token)
    res =await db.execute(query)
    user_token = res.scalar_one_or_none()
    if not user_token:
        return None
    query = select(User).where(User.id == user_token.user_id)
    res =await db.execute(query)
    user = res.scalar_one_or_none()
    return user

async def change_user_password(db:AsyncSession,user:User,old_password:str,new_passowrd:str):
    hashed_password = get_hashed_password(user.password)
    query = select(User).where(User.password == hashed_password)
    res =await db.execute(query)
    user = res.scalar_one_or_none()
    if  not user :
        return None
    hashed_new_password = get_hashed_password(new_passowrd)
    user.password = hashed_new_password
    await db.add(user)
    await db.commit()
    await db.refresh(user)
    return True