"""
交易模块测试

包含币安量化订单系统的所有测试用例
"""

# 测试模块导入
from .test_binance_quant_order_manager import TestBinanceQuantOrderManager
from .test_order_router import TestTradingModeRouter, TestEnvironmentConfirmation
from .test_paper_trading_engine import TestPaperTradingEngine
from .test_order_sync_manager import TestOrderSyncManager
from .test_integration import TestTradingSystemIntegration

__all__ = [
    "TestBinanceQuantOrderManager",
    "TestTradingModeRouter", 
    "TestEnvironmentConfirmation",
    "TestPaperTradingEngine",
    "TestOrderSyncManager",
    "TestTradingSystemIntegration"
]