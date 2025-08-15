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
from src.utils.logger import get_logger, setup_logging

# Use uvloop for better async performance on Unix systems
if sys.platform != "win32":
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Setup logging
setup_logging()
logger = get_logger(__name__)

# FastAPI app for web interface
app = FastAPI(
    title="Crypto Quant Trading System",
    description="Lightweight cryptocurrency quantitative trading system",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TradingSystem:
    """Main trading system orchestrator."""
    
    def __init__(self):
        self.running = False
        self.tasks = []
        
    async def start(self):
        """Start the trading system."""
        logger.info("Starting trading system", mode=settings.trading_mode)
        
        # Initialize database
        init_database()
        logger.info("Database initialized")
        
        # Start components based on trading mode
        if settings.trading_mode == "backtest":
            await self.start_backtest()
        elif settings.trading_mode == "paper":
            await self.start_paper_trading()
        elif settings.trading_mode == "live":
            await self.start_live_trading()
        
        self.running = True
        logger.info("Trading system started")
    
    async def start_backtest(self):
        """Start backtesting mode."""
        logger.info("Starting backtest mode")
        # TODO: Implement backtest logic
    
    async def start_paper_trading(self):
        """Start paper trading mode."""
        logger.info("Starting paper trading mode")
        # TODO: Implement paper trading logic
    
    async def start_live_trading(self):
        """Start live trading mode."""
        logger.warning("Starting LIVE trading mode - Real money at risk!")
        # TODO: Implement live trading logic
    
    async def stop(self):
        """Stop the trading system."""
        logger.info("Stopping trading system")
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        logger.info("Trading system stopped")
    
    def handle_signal(self, signum, frame):
        """Handle system signals for graceful shutdown."""
        logger.info(f"Received signal {signum}, initiating shutdown")
        asyncio.create_task(self.stop())


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


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Crypto Quant Trading System",
        "version": "0.1.0",
        "mode": settings.trading_mode,
        "status": "running" if trading_system.running else "stopped",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@click.group()
def cli():
    """Crypto Quant Trading System CLI."""
    pass


@cli.command()
@click.option("--mode", type=click.Choice(["paper", "backtest", "live"]), default="paper")
def trade(mode: str):
    """Start the trading system."""
    settings.trading_mode = mode
    
    async def run():
        # Setup signal handlers
        signal.signal(signal.SIGINT, trading_system.handle_signal)
        signal.signal(signal.SIGTERM, trading_system.handle_signal)
        
        # Start trading system
        await trading_system.start()
        
        # Keep running until stopped
        while trading_system.running:
            await asyncio.sleep(1)
    
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        logger.info("Trading system exited")


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