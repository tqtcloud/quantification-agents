#!/usr/bin/env python3
import asyncio
import signal
import sys
from typing import Optional

import click
import uvloop
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.core.database import init_database
from src.core.system_orchestrator import system_orchestrator
from src.core.config_manager import config_manager
from src.monitoring.system_monitor import system_monitor
from src.monitoring.alert_manager import alert_manager
from src.utils.logger import get_logger, setup_logging
from src.web.api import app as web_app

# Use uvloop for better async performance on Unix systems
if sys.platform != "win32":
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Setup logging
setup_logging()
logger = get_logger(__name__)

# 使用增强的 web app
app = web_app


class TradingSystem:
    """Main trading system orchestrator."""
    
    def __init__(self):
        self.running = False
        self.components_initialized = False
        
    async def initialize_components(self):
        """初始化系统组件"""
        if self.components_initialized:
            return
            
        logger.info("初始化系统组件")
        
        # 初始化数据库
        init_database()
        
        # 注册核心组件到系统编排器
        system_orchestrator.register_component(
            "config_manager", 
            config_manager,
            startup_func=config_manager.start_watching,
            shutdown_func=config_manager.stop_watching
        )
        
        system_orchestrator.register_component(
            "system_monitor", 
            system_monitor,
            dependencies=["config_manager"],
            startup_func=lambda: system_monitor.start_monitoring(),
            shutdown_func=system_monitor.stop_monitoring
        )
        
        system_orchestrator.register_component(
            "alert_manager", 
            alert_manager,
            dependencies=["system_monitor"]
        )
        
        # 设置监控和告警的联动
        system_monitor.add_metric_callback(alert_manager.check_alerts)
        
        self.components_initialized = True
        logger.info("系统组件初始化完成")
        
    async def start(self):
        """Start the trading system."""
        logger.info("启动交易系统", mode=settings.trading_mode)
        
        try:
            # 初始化组件
            await self.initialize_components()
            
            # 启动所有组件
            await system_orchestrator.start_all()
            
            # 根据交易模式启动特定逻辑
            if settings.trading_mode == "backtest":
                await self.start_backtest()
            elif settings.trading_mode == "paper":
                await self.start_paper_trading()
            elif settings.trading_mode == "live":
                await self.start_live_trading()
            
            self.running = True
            logger.info("交易系统启动成功")
            
        except Exception as e:
            logger.error(f"交易系统启动失败: {e}")
            await self.stop()
            raise
    
    async def start_backtest(self):
        """Start backtesting mode."""
        logger.info("启动回测模式")
        # TODO: 实现回测逻辑
    
    async def start_paper_trading(self):
        """Start paper trading mode."""
        logger.info("启动模拟交易模式")
        # TODO: 实现模拟交易逻辑
    
    async def start_live_trading(self):
        """Start live trading mode."""
        logger.warning("启动实盘交易模式 - 真实资金有风险!")
        # TODO: 实现实盘交易逻辑
    
    async def stop(self):
        """Stop the trading system."""
        logger.info("停止交易系统")
        self.running = False
        
        try:
            # 使用系统编排器优雅关闭所有组件
            await system_orchestrator.stop_all()
            logger.info("交易系统已停止")
        except Exception as e:
            logger.error(f"停止交易系统时出错: {e}")
    
    def get_status(self):
        """获取系统状态"""
        return {
            "running": self.running,
            "mode": settings.trading_mode,
            "components": system_orchestrator.get_system_status(),
            "health_score": system_monitor.get_system_health_score() if system_monitor._monitoring else 0
        }


trading_system = TradingSystem()


@app.on_event("startup")
async def startup_event():
    """FastAPI startup event."""
    logger.info("FastAPI application starting")


@app.on_event("shutdown")
async def shutdown_event():
    """FastAPI shutdown event."""
    logger.info("FastAPI application shutting down")
    await trading_system.stop()


@app.get("/system/status")
async def system_status():
    """系统状态端点"""
    return trading_system.get_status()


@app.get("/system/metrics")
async def system_metrics():
    """系统指标端点"""
    return system_monitor.get_current_metrics()


@app.get("/system/alerts")
async def system_alerts():
    """系统告警端点"""
    return {
        "active_alerts": [alert.to_dict() for alert in alert_manager.get_active_alerts()],
        "stats": alert_manager.get_alert_stats()
    }


@click.group()
def cli():
    """Crypto Quant Trading System CLI."""
    pass


@cli.command()
@click.option("--mode", type=click.Choice(["paper", "backtest", "live"]), default="paper")
@click.option("--env", type=click.Choice(["development", "testing", "production"]), default="development")
def trade(mode: str, env: str):
    """Start the trading system."""
    settings.trading_mode = mode
    config_manager.switch_environment(env)
    
    async def run():
        try:
            # Start trading system with orchestrator
            await trading_system.start()
            
            # Wait for shutdown signal
            await system_orchestrator.wait_for_shutdown()
            
        except KeyboardInterrupt:
            logger.info("收到键盘中断信号")
        finally:
            await trading_system.stop()
    
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("收到键盘中断")
    finally:
        logger.info("交易系统已退出")


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
def web(host: str, port: int, reload: bool):
    """Start the web interface."""
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=settings.log_level.lower(),
    )


@cli.command()
@click.option("--strategy", required=True, help="Strategy to backtest")
@click.option("--start-date", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end-date", required=True, help="End date (YYYY-MM-DD)")
@click.option("--initial-capital", default=10000, help="Initial capital")
def backtest(strategy: str, start_date: str, end_date: str, initial_capital: float):
    """Run a backtest."""
    logger.info(
        "Running backtest",
        strategy=strategy,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
    )
    # TODO: Implement backtest command


if __name__ == "__main__":
    cli()