"""FastAPI应用主入口"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
from typing import Optional
import logging

from .models import APIInfoResponse, ErrorResponse, TradingMode
from .monitoring_api import MonitoringAPI
from .websocket_handler import WebSocketHandler

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="Crypto Quantitative Trading System API",
    description="加密货币量化交易系统监控API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该指定具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化子系统
monitoring_api = MonitoringAPI()
websocket_handler = WebSocketHandler()

# 全局异常处理器
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error_code="INTERNAL_ERROR",
            error_message=str(exc),
            timestamp=datetime.now()
        ).dict()
    )

# 健康检查端点
@app.get("/health")
async def health_check():
    """系统健康检查"""
    return {"status": "healthy", "timestamp": datetime.now()}

# API信息端点
@app.get("/api/info", response_model=APIInfoResponse)
async def get_api_info():
    """获取API信息"""
    return APIInfoResponse(
        version="1.0.0",
        name="Crypto Quantitative Trading System API",
        description="加密货币量化交易系统监控和控制API",
        docs_url="/api/docs",
        health_check_url="/health",
        websocket_url="/ws",
        supported_trading_modes=[TradingMode.PAPER, TradingMode.LIVE],
        supported_exchanges=["binance"]
    )

# 包含监控API路由
app.include_router(monitoring_api.router, prefix="/api", tags=["monitoring"])

# 包含WebSocket路由
app.include_router(websocket_handler.router, prefix="/ws", tags=["websocket"])

# 启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化"""
    logger.info("正在启动量化交易系统API...")
    
    # 初始化监控API
    await monitoring_api.initialize()
    
    # 初始化WebSocket处理器
    await websocket_handler.initialize()
    
    logger.info("量化交易系统API启动完成")

# 关闭事件
@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时的清理"""
    logger.info("正在关闭量化交易系统API...")
    
    # 清理监控API
    await monitoring_api.cleanup()
    
    # 清理WebSocket处理器
    await websocket_handler.cleanup()
    
    logger.info("量化交易系统API已关闭")

def run_server(host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
    """运行FastAPI服务器"""
    uvicorn.run(
        "src.web.api:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug"
    )

if __name__ == "__main__":
    run_server(debug=True)