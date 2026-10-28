"""Bounded HTTP adapter around the selected complete MLflow pyfunc."""
from contextlib import asynccontextmanager
from datetime import datetime,timezone
import json
import logging
import threading
import time
import anyio
import numpy as np
import pandas as pd
import psutil
from fastapi import FastAPI,HTTPException,Request
from starlette.responses import JSONResponse
from .settings import Settings,MODEL_NAME,MODEL_VERSION,RELEASE_ID
from .model_loader import load_model
from .schemas import RecommendRequest,RecommendResponse,Recommendation,UserRecommendations,ModelIdentity

logger=logging.getLogger('recsys.http')
ENDPOINTS={'/health/live','/health/ready','/model-info','/recommend','/openapi.json','/docs','/redoc'}


class RequestMiddleware:
    """Bound body buffering before JSON parsing, plus sanitized ASGI access logs."""
    def __init__(self,app,max_bytes):self.app,self.max_bytes=app,max_bytes

    async def __call__(self,scope,receive,send):
        if scope['type']!='http':return await self.app(scope,receive,send)
        start=time.perf_counter();status=500
        async def record_send(message):
            nonlocal status
            if message['type']=='http.response.start':status=message['status']
            await send(message)
        try:
            if scope['method']=='POST' and scope['path']=='/recommend':
                headers=dict(scope.get('headers',[]))
                try:length=int(headers.get(b'content-length',b'0'))
                except ValueError:
                    scope['error_category']='invalid_content_length'
                    return await JSONResponse({'detail':'Invalid content length'},400)(scope,receive,record_send)
                if length<0 or length>self.max_bytes:
                    scope['error_category']='body_limit'
                    return await JSONResponse({'detail':'Request body too large'},413)(scope,receive,record_send)
                chunks=[];size=0
                while True:
                    message=await receive()
                    if message['type']=='http.disconnect':return
                    chunk=message.get('body',b'');size+=len(chunk)
                    if size>self.max_bytes:
                        scope['error_category']='body_limit'
                        return await JSONResponse({'detail':'Request body too large'},413)(scope,receive,record_send)
                    chunks.append(chunk)
                    if not message.get('more_body',False):break
                pending=True
                async def buffered_receive():
                    nonlocal pending
                    if pending:
                        pending=False
                        return {'type':'http.request','body':b''.join(chunks),'more_body':False}
                    return await receive()
                await self.app(scope,buffered_receive,record_send)
            else:await self.app(scope,receive,record_send)
        finally:
            logger.info(json.dumps(dict(timestamp=datetime.now(timezone.utc).isoformat(),
                endpoint=scope['path'] if scope['path'] in ENDPOINTS else 'unknown',status=status,
                duration_ms=round((time.perf_counter()-start)*1000,3),batch_size=scope.get('batch_size'),
                model_version=MODEL_VERSION,release_id=RELEASE_ID,
                error_category=scope.get('error_category','validation' if status==422 else None))))


def response_from_predictions(frame,request,identity):
    if list(frame.columns)!=['userId','movieId','rank','score','fallback'] or frame.duplicated(['userId','movieId']).any():
        raise ValueError('Invalid prediction schema or duplicate recommendation')
    if not frame.userId.isin(request.user_ids).all() or not np.isfinite(frame.score.to_numpy()).all():
        raise ValueError('Unexpected user or nonfinite prediction')
    present=set(frame.userId)
    if frame.userId.drop_duplicates().tolist()!=[u for u in request.user_ids if u in present]:
        raise ValueError('Prediction query ordering changed')
    results=[]
    for user in request.user_ids:
        rows=frame[frame.userId.eq(user)]
        if len(rows)>request.top_k or rows['rank'].tolist()!=list(range(1,len(rows)+1)):
            raise ValueError('Invalid prediction ranks or Top-K')
        recommendations=[Recommendation(movie_id=int(row.movieId),rank=int(row.rank),score=float(row.score),fallback=str(row.fallback))
                         for row in rows.itertuples(index=False)]
        results.append(UserRecommendations(user_id=user,recommendations=recommendations))
    return RecommendResponse(model=ModelIdentity(**identity),results=results)


def create_app(settings=None,loader=load_model):
    settings=settings or Settings.from_env()
    @asynccontextmanager
    async def lifespan(app):
        started=time.perf_counter()
        try:loaded=await anyio.to_thread.run_sync(loader,settings)
        except Exception as error:
            logger.error(json.dumps(dict(event='model_load_failed',error_category=type(error).__name__)))
            raise RuntimeError('Configured model failed validation or loading') from None
        app.state.loaded=loaded;app.state.load_count+=1;app.state.ready=True
        app.state.startup_seconds=time.perf_counter()-started
        logger.info(json.dumps(dict(event='model_ready',model_version=MODEL_VERSION,release_id=loaded.info['release_id'],
            startup_seconds=app.state.startup_seconds,load_count=app.state.load_count,rss_bytes=psutil.Process().memory_info().rss)))
        try:yield
        finally:app.state.ready=False;app.state.loaded=None
    app=FastAPI(title='MovieLens Hybrid Recommender',version='phase6',lifespan=lifespan,debug=False)
    app.state.loaded=None;app.state.ready=False;app.state.load_count=0;app.state.startup_seconds=None
    app.state.lock=threading.Lock();app.state.requests_served=0
    app.add_middleware(RequestMiddleware,max_bytes=settings.max_body_bytes)

    @app.get('/health/live')
    async def live():return {'status':'alive'}

    @app.get('/health/ready')
    async def ready():
        if not app.state.ready:raise HTTPException(503,'Model not ready')
        return {'status':'ready','model_loaded':True}

    @app.get('/model-info')
    async def info():
        if not app.state.ready:raise HTTPException(503,'Model not ready')
        return dict(app.state.loaded.info,load_count=app.state.load_count,startup_seconds=app.state.startup_seconds,
                    requests_served=app.state.requests_served,rss_bytes=psutil.Process().memory_info().rss)

    @app.post('/recommend',response_model=RecommendResponse)
    async def recommend(payload:RecommendRequest,request:Request):
        request.scope['batch_size']=len(payload.user_ids)
        if len(payload.user_ids)>settings.max_batch_users:
            request.scope['error_category']='batch_limit';raise HTTPException(413,'User batch exceeds configured limit')
        if not app.state.ready:raise HTTPException(503,'Model not ready')
        if not app.state.lock.acquire(blocking=False):
            request.scope['error_category']='busy';raise HTTPException(503,'Inference busy; retry later',headers={'Retry-After':'1'})
        try:
            inputs=pd.DataFrame({'userId':payload.user_ids,'top_k':payload.top_k},dtype='int64')
            frame=await anyio.to_thread.run_sync(app.state.loaded.model.predict,inputs)
            identity={key:app.state.loaded.info[key] for key in ['name','version','release_status']}
            result=response_from_predictions(frame,payload,identity);app.state.requests_served+=1
            return result
        except MemoryError:
            request.scope['error_category']='resource_exhausted';raise HTTPException(503,'Inference resource limit reached') from None
        except Exception as error:
            request.scope['error_category']='prediction_failure'
            logger.error(json.dumps(dict(event='prediction_failed',error_category=type(error).__name__)))
            raise HTTPException(500,'Model prediction failed') from None
        finally:app.state.lock.release()
    return app
