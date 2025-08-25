"""
量化交易API服务入口点

这个模块提供了API服务的主要入口点，可以直接运行或通过uvicorn启动。
支持开发模式和生产模式的不同配置。
"""

import os
import sys
import asyncio
from typing import Optional

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from src.api.trading_api import TradingAPI
from src.config import Config
from src.strategy.strategy_manager import StrategyManager
from src.strategy.signal_aggregator import SignalAggregator
import structlog

logger = structlog.get_logger(__name__)

# 全局API实例
api_instance: Optional[TradingAPI] = None


def create_app() -> TradingAPI:
    """创建API应用实例"""
    global api_instance
    
    if api_instance is None:
        # 加载配置
        config = Config.load_from_env()
        
        # 创建API实例
        api_instance = TradingAPI(config)
        
        # 这里可以注入依赖的服务
        # 在实际部署中，应该从服务注册中心或依赖注入容器获取这些服务
        try:
            # 注入策略管理器（如果可用）
            if hasattr(config, 'strategy_manager_enabled') and config.strategy_manager_enabled:
                strategy_manager = StrategyManager(config)
                api_instance.set_strategy_manager(strategy_manager)
            
            # 注入信号聚合器（如果可用）
            if hasattr(config, 'signal_aggregator_enabled') and config.signal_aggregator_enabled:
                signal_aggregator = SignalAggregator(config)
                api_instance.set_signal_aggregator(signal_aggregator)
                
        except Exception as e:
            logger.warning(f"Failed to inject services: {e}")
            # 在开发和测试环境中继续运行，不注入服务
        
        logger.info("API application created successfully")
    
    return api_instance


def get_app():
    """获取FastAPI应用实例（用于uvicorn）"""
    api = create_app()
    return api.app


# 为uvicorn创建应用实例
app = get_app()


async def start_api_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    reload: bool = False,
    debug: bool = False
):
    """启动API服务器"""
    try:
        logger.info(f"Starting Trading API server on {host}:{port}")
        
        api = create_app()
        
        # 运行服务器
        api.run(
            host=host,
            port=port,
            workers=workers,
            reload=reload
        )
        
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        raise


def main():
    """主函数 - 用于直接运行"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quantification Trading API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (development)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # 从环境变量覆盖参数
    host = os.getenv("API_HOST", args.host)
    port = int(os.getenv("API_PORT", args.port))
    workers = int(os.getenv("API_WORKERS", args.workers))
    reload = os.getenv("API_RELOAD", "false").lower() == "true" or args.reload
    debug = os.getenv("API_DEBUG", "false").lower() == "true" or args.debug
    
    try:
        asyncio.run(start_api_server(
            host=host,
            port=port,
            workers=workers,
            reload=reload,
            debug=debug
        ))
    except KeyboardInterrupt:
        logger.info("API server stopped by user")
    except Exception as e:
        logger.error(f"API server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()