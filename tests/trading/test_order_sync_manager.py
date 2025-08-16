import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from src.trading.order_sync_manager import (
    OrderSyncManager, OrderSyncRecord, DataIsolationConfig, SyncStatus
)
from src.trading.order_router import TradingMode
from src.core.models import Order, OrderSide, OrderType, OrderStatus


class TestOrderSyncManager:
    """订单同步管理器测试"""

    @pytest.fixture
    def isolation_config(self):
        """数据隔离配置"""
        return DataIsolationConfig(
            enable_isolation=True,
            paper_data_prefix="paper_",
            live_data_prefix="live_",
            isolation_level="STRICT"
        )

    @pytest.fixture
    async def sync_manager(self, isolation_config):
        """同步管理器实例"""
        manager = OrderSyncManager(isolation_config)
        await manager.initialize()
        return manager

    @pytest.fixture
    def sample_order(self):
        """示例订单"""
        return Order(
            order_id="12345",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,
            price=50000.0,
            status=OrderStatus.NEW,
            client_order_id="test_order_001",
            created_at=datetime.utcnow()
        )

    async def test_initialization(self, isolation_config):
        """测试初始化"""
        manager = OrderSyncManager(isolation_config)
        
        # 初始化前状态
        assert not manager._running
        assert manager._sync_task is None
        assert len(manager.sync_records) == 0
        
        await manager.initialize()
        
        # 初始化后状态
        assert manager._running
        assert manager._sync_task is not None
        assert not manager._sync_task.done()

    async def test_register_order_paper(self, sync_manager, sample_order):
        """测试注册模拟盘订单"""
        await sync_manager.register_order(
            order=sample_order,
            trading_mode=TradingMode.PAPER,
            execution_result={"status": "FILLED"}
        )
        
        # 验证订单存储在模拟盘
        assert sample_order.client_order_id in sync_manager.paper_orders
        assert sample_order.client_order_id not in sync_manager.live_orders
        
        # 验证同步记录创建
        assert sample_order.client_order_id in sync_manager.sync_records
        sync_record = sync_manager.sync_records[sample_order.client_order_id]
        assert sync_record.trading_mode == TradingMode.PAPER
        assert sync_record.local_status == OrderStatus.NEW
        
        # 验证统计更新
        assert sync_manager.isolation_stats["paper_operations"] == 1
        assert sync_manager.isolation_stats["live_operations"] == 0

    async def test_register_order_live(self, sync_manager, sample_order):
        """测试注册实盘订单"""
        await sync_manager.register_order(
            order=sample_order,
            trading_mode=TradingMode.LIVE,
            execution_result={"orderId": 12345}
        )
        
        # 验证订单存储在实盘
        assert sample_order.client_order_id in sync_manager.live_orders
        assert sample_order.client_order_id not in sync_manager.paper_orders
        
        # 验证同步记录
        sync_record = sync_manager.sync_records[sample_order.client_order_id]
        assert sync_record.trading_mode == TradingMode.LIVE
        
        # 验证统计更新
        assert sync_manager.isolation_stats["live_operations"] == 1

    async def test_update_order_status_success(self, sync_manager, sample_order):
        """测试更新订单状态成功"""
        # 先注册订单
        await sync_manager.register_order(sample_order, TradingMode.PAPER)
        
        # 更新状态
        remote_data = {"status": "FILLED", "executedQty": "0.001"}
        await sync_manager.update_order_status(
            client_order_id=sample_order.client_order_id,
            new_status=OrderStatus.FILLED,
            trading_mode=TradingMode.PAPER,
            remote_data=remote_data
        )
        
        # 验证本地订单状态更新
        order = sync_manager.paper_orders[sample_order.client_order_id]
        assert order.status == OrderStatus.FILLED
        assert order.updated_at is not None
        
        # 验证同步记录更新
        sync_record = sync_manager.sync_records[sample_order.client_order_id]
        assert sync_record.local_status == OrderStatus.FILLED
        assert sync_record.remote_status == OrderStatus.FILLED
        assert sync_record.sync_status == SyncStatus.IN_SYNC

    async def test_update_order_status_cross_environment_violation(self, sync_manager, sample_order):
        """测试跨环境访问违规"""
        # 注册为模拟盘订单
        await sync_manager.register_order(sample_order, TradingMode.PAPER)
        
        # 尝试以实盘模式更新（应该失败）
        with pytest.raises(PermissionError, match="Cross-environment access denied"):
            await sync_manager.update_order_status(
                client_order_id=sample_order.client_order_id,
                new_status=OrderStatus.FILLED,
                trading_mode=TradingMode.LIVE
            )
        
        # 验证违规统计
        assert sync_manager.isolation_stats["isolation_violations"] == 1

    async def test_get_order_by_environment(self, sync_manager, sample_order):
        """测试按环境获取订单"""
        # 注册模拟盘订单
        await sync_manager.register_order(sample_order, TradingMode.PAPER)
        
        # 从模拟盘获取
        order = await sync_manager.get_order(
            sample_order.client_order_id, 
            TradingMode.PAPER
        )
        assert order is not None
        assert order.symbol == "BTCUSDT"
        
        # 从实盘获取（应该为空）
        order = await sync_manager.get_order(
            sample_order.client_order_id,
            TradingMode.LIVE
        )
        assert order is None

    async def test_get_order_cross_environment(self, sync_manager, sample_order):
        """测试跨环境查询"""
        # 注册模拟盘订单
        await sync_manager.register_order(sample_order, TradingMode.PAPER)
        
        # 跨环境查询（不指定trading_mode）
        order = await sync_manager.get_order(sample_order.client_order_id)
        assert order is not None
        assert order.symbol == "BTCUSDT"
        
        # 验证跨环境查询统计
        assert sync_manager.isolation_stats["cross_environment_queries"] == 1

    async def test_get_orders_by_environment(self, sync_manager):
        """测试按环境获取订单列表"""
        # 创建多个订单
        orders = []
        for i in range(3):
            order = Order(
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=0.001,
                client_order_id=f"test_order_{i}"
            )
            orders.append(order)
        
        # 注册到不同环境
        await sync_manager.register_order(orders[0], TradingMode.PAPER)
        await sync_manager.register_order(orders[1], TradingMode.PAPER)
        await sync_manager.register_order(orders[2], TradingMode.LIVE)
        
        # 获取模拟盘订单
        paper_orders = await sync_manager.get_orders_by_environment(TradingMode.PAPER)
        assert len(paper_orders) == 2
        
        # 获取实盘订单
        live_orders = await sync_manager.get_orders_by_environment(TradingMode.LIVE)
        assert len(live_orders) == 1

    async def test_archive_completed_order(self, sync_manager, sample_order):
        """测试归档已完成订单"""
        # 注册并完成订单
        await sync_manager.register_order(sample_order, TradingMode.PAPER)
        sample_order.status = OrderStatus.FILLED
        
        # 归档订单
        await sync_manager.archive_completed_order(
            sample_order.client_order_id,
            TradingMode.PAPER
        )
        
        # 验证订单从活跃列表移除
        assert sample_order.client_order_id not in sync_manager.paper_orders
        
        # 验证历史记录创建
        assert len(sync_manager.paper_order_history) == 1
        history_record = sync_manager.paper_order_history[0]
        assert history_record["client_order_id"] == sample_order.client_order_id
        assert history_record["trading_mode"] == "paper"
        assert "archived_at" in history_record
        
        # 验证同步记录清理
        assert sample_order.client_order_id not in sync_manager.sync_records

    async def test_get_order_history(self, sync_manager, sample_order):
        """测试获取订单历史"""
        # 注册并归档订单
        await sync_manager.register_order(sample_order, TradingMode.PAPER)
        await sync_manager.archive_completed_order(
            sample_order.client_order_id,
            TradingMode.PAPER
        )
        
        # 获取历史记录
        history = await sync_manager.get_order_history(TradingMode.PAPER)
        assert len(history) == 1
        assert history[0]["symbol"] == "BTCUSDT"
        
        # 测试符号过滤
        history_filtered = await sync_manager.get_order_history(
            TradingMode.PAPER,
            symbol="BTCUSDT"
        )
        assert len(history_filtered) == 1
        
        # 测试不匹配的符号
        history_empty = await sync_manager.get_order_history(
            TradingMode.PAPER,
            symbol="ETHUSDT"
        )
        assert len(history_empty) == 0

    async def test_get_order_history_limit(self, sync_manager):
        """测试订单历史限制"""
        # 创建多个历史记录
        for i in range(5):
            sync_manager.paper_order_history.append({
                "client_order_id": f"hist_{i}",
                "symbol": "BTCUSDT"
            })
        
        # 测试限制
        history = await sync_manager.get_order_history(TradingMode.PAPER, limit=3)
        assert len(history) == 3
        
        # 测试无限制
        history_all = await sync_manager.get_order_history(TradingMode.PAPER, limit=0)
        assert len(history_all) == 5

    async def test_data_access_permission_strict(self, sync_manager, sample_order):
        """测试严格隔离模式的数据访问权限"""
        # 注册模拟盘订单
        await sync_manager.register_order(sample_order, TradingMode.PAPER)
        
        # 模拟盘访问模拟盘订单（应该允许）
        has_permission = sync_manager._check_data_access_permission(
            sample_order.client_order_id,
            TradingMode.PAPER
        )
        assert has_permission is True
        
        # 实盘访问模拟盘订单（应该拒绝）
        has_permission = sync_manager._check_data_access_permission(
            sample_order.client_order_id,
            TradingMode.LIVE
        )
        assert has_permission is False

    async def test_data_access_permission_moderate(self, isolation_config, sample_order):
        """测试中等隔离模式"""
        isolation_config.isolation_level = "MODERATE"
        sync_manager = OrderSyncManager(isolation_config)
        await sync_manager.initialize()
        
        await sync_manager.register_order(sample_order, TradingMode.PAPER)
        
        # 中等模式允许跨环境读取
        has_permission = sync_manager._check_data_access_permission(
            sample_order.client_order_id,
            TradingMode.LIVE
        )
        assert has_permission is True

    async def test_data_access_permission_loose(self, isolation_config, sample_order):
        """测试宽松隔离模式"""
        isolation_config.isolation_level = "LOOSE"
        sync_manager = OrderSyncManager(isolation_config)
        await sync_manager.initialize()
        
        await sync_manager.register_order(sample_order, TradingMode.PAPER)
        
        # 宽松模式允许所有访问
        has_permission = sync_manager._check_data_access_permission(
            sample_order.client_order_id,
            TradingMode.LIVE
        )
        assert has_permission is True

    async def test_data_access_permission_disabled(self, isolation_config, sample_order):
        """测试禁用隔离"""
        isolation_config.enable_isolation = False
        sync_manager = OrderSyncManager(isolation_config)
        await sync_manager.initialize()
        
        await sync_manager.register_order(sample_order, TradingMode.PAPER)
        
        # 禁用隔离允许所有访问
        has_permission = sync_manager._check_data_access_permission(
            sample_order.client_order_id,
            TradingMode.LIVE
        )
        assert has_permission is True

    async def test_sync_statistics(self, sync_manager, sample_order):
        """测试同步统计"""
        # 注册几个订单
        await sync_manager.register_order(sample_order, TradingMode.PAPER)
        
        order2 = Order(
            symbol="ETHUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.01,
            client_order_id="test_order_002"
        )
        await sync_manager.register_order(order2, TradingMode.LIVE)
        
        # 更新一个订单状态为同步
        sync_record = sync_manager.sync_records[sample_order.client_order_id]
        sync_record.sync_status = SyncStatus.IN_SYNC
        
        # 获取统计
        stats = sync_manager.get_sync_statistics()
        
        assert stats["total_records"] == 2
        assert stats["in_sync_records"] == 1
        assert stats["out_of_sync_records"] == 0
        assert stats["paper_orders_count"] == 1
        assert stats["live_orders_count"] == 1
        assert stats["sync_rate"] == 0.5
        assert "isolation_stats" in stats

    async def test_environment_summary(self, sync_manager, sample_order):
        """测试环境摘要"""
        # 注册订单到不同环境
        await sync_manager.register_order(sample_order, TradingMode.PAPER)
        
        order2 = Order(
            symbol="ETHUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.01,
            client_order_id="test_order_002"
        )
        await sync_manager.register_order(order2, TradingMode.LIVE)
        
        # 归档一个订单
        await sync_manager.archive_completed_order(
            sample_order.client_order_id,
            TradingMode.PAPER
        )
        
        summary = sync_manager.get_environment_summary()
        
        assert summary["paper_trading"]["active_orders"] == 0
        assert summary["paper_trading"]["completed_orders"] == 1
        assert summary["live_trading"]["active_orders"] == 1
        assert summary["live_trading"]["completed_orders"] == 0
        assert summary["isolation_config"]["enabled"] is True
        assert summary["isolation_config"]["level"] == "STRICT"

    async def test_force_sync_order(self, sync_manager, sample_order):
        """测试强制同步订单"""
        # 注册订单
        await sync_manager.register_order(sample_order, TradingMode.PAPER)
        
        # 强制同步
        success = await sync_manager.force_sync_order(sample_order.client_order_id)
        
        assert success is True
        
        # 验证同步记录更新
        sync_record = sync_manager.sync_records[sample_order.client_order_id]
        assert sync_record.sync_status == SyncStatus.IN_SYNC
        assert sync_record.sync_attempts == 1

    async def test_force_sync_order_not_found(self, sync_manager):
        """测试强制同步不存在的订单"""
        success = await sync_manager.force_sync_order("nonexistent_order")
        assert success is False

    async def test_reset_environment_data_requires_confirmation(self, sync_manager, sample_order):
        """测试重置环境数据需要确认"""
        await sync_manager.register_order(sample_order, TradingMode.PAPER)
        
        # 不确认应该失败
        with pytest.raises(ValueError, match="requires explicit confirmation"):
            await sync_manager.reset_environment_data(TradingMode.PAPER, confirm=False)

    async def test_reset_environment_data_paper(self, sync_manager, sample_order):
        """测试重置模拟盘环境数据"""
        # 添加数据
        await sync_manager.register_order(sample_order, TradingMode.PAPER)
        await sync_manager.archive_completed_order(
            sample_order.client_order_id,
            TradingMode.PAPER
        )
        
        assert len(sync_manager.paper_orders) == 0  # 已归档
        assert len(sync_manager.paper_order_history) == 1
        
        # 重置
        await sync_manager.reset_environment_data(TradingMode.PAPER, confirm=True)
        
        # 验证数据清理
        assert len(sync_manager.paper_orders) == 0
        assert len(sync_manager.paper_order_history) == 0
        assert len(sync_manager.sync_records) == 0

    async def test_reset_environment_data_live(self, sync_manager, sample_order):
        """测试重置实盘环境数据"""
        await sync_manager.register_order(sample_order, TradingMode.LIVE)
        
        await sync_manager.reset_environment_data(TradingMode.LIVE, confirm=True)
        
        assert len(sync_manager.live_orders) == 0
        assert len(sync_manager.live_order_history) == 0

    async def test_export_environment_data_paper(self, sync_manager, sample_order):
        """测试导出模拟盘环境数据"""
        await sync_manager.register_order(sample_order, TradingMode.PAPER)
        
        # 归档一个订单创建历史
        await sync_manager.archive_completed_order(
            sample_order.client_order_id,
            TradingMode.PAPER
        )
        
        export_data = sync_manager.export_environment_data(TradingMode.PAPER)
        
        assert export_data["trading_mode"] == "paper"
        assert len(export_data["active_orders"]) == 0  # 已归档
        assert len(export_data["order_history"]) == 1
        assert "exported_at" in export_data

    async def test_export_environment_data_live(self, sync_manager, sample_order):
        """测试导出实盘环境数据"""
        await sync_manager.register_order(sample_order, TradingMode.LIVE)
        
        export_data = sync_manager.export_environment_data(TradingMode.LIVE)
        
        assert export_data["trading_mode"] == "live"
        assert len(export_data["active_orders"]) == 1
        assert len(export_data["order_history"]) == 0

    async def test_sync_loop_operation(self, sync_manager, sample_order):
        """测试同步循环操作"""
        # 注册订单
        await sync_manager.register_order(sample_order, TradingMode.PAPER)
        
        # 设置订单为需要同步
        sync_record = sync_manager.sync_records[sample_order.client_order_id]
        sync_record.sync_status = SyncStatus.OUT_OF_SYNC
        
        # 手动触发一次同步
        await sync_manager._perform_sync()
        
        # 验证同步尝试
        assert sync_record.sync_attempts > 0
        assert sync_record.last_sync_time is not None

    async def test_sync_error_handling(self, sync_manager, sample_order):
        """测试同步错误处理"""
        await sync_manager.register_order(sample_order, TradingMode.PAPER)
        
        sync_record = sync_manager.sync_records[sample_order.client_order_id]
        sync_record.sync_status = SyncStatus.OUT_OF_SYNC
        
        # 模拟同步失败
        with patch.object(sync_manager, '_sync_single_record', 
                         side_effect=Exception("Sync failed")):
            await sync_manager._perform_sync()
        
        # 验证错误记录
        assert sync_record.sync_status == SyncStatus.ERROR
        assert sync_record.error_message == "Sync failed"
        assert sync_record.sync_attempts == 1

    async def test_cleanup(self, sync_manager, sample_order):
        """测试清理"""
        await sync_manager.register_order(sample_order, TradingMode.PAPER)
        
        # 确认同步任务在运行
        assert sync_manager._running is True
        assert sync_manager._sync_task is not None
        
        await sync_manager.cleanup()
        
        # 验证清理
        assert sync_manager._running is False
        assert sync_manager._sync_task.done()

    async def test_order_sync_record_creation(self, sample_order):
        """测试订单同步记录创建"""
        sync_record = OrderSyncRecord(
            order_id=sample_order.order_id,
            client_order_id=sample_order.client_order_id,
            symbol=sample_order.symbol,
            trading_mode=TradingMode.PAPER,
            local_status=sample_order.status
        )
        
        assert sync_record.order_id == sample_order.order_id
        assert sync_record.trading_mode == TradingMode.PAPER
        assert sync_record.sync_status == SyncStatus.PENDING  # 默认值
        assert sync_record.sync_attempts == 0

    async def test_data_isolation_config_defaults(self):
        """测试数据隔离配置默认值"""
        config = DataIsolationConfig()
        
        assert config.enable_isolation is True
        assert config.paper_data_prefix == "paper_"
        assert config.live_data_prefix == "live_"
        assert "system_config" in config.shared_data_keys
        assert "market_data" in config.shared_data_keys
        assert config.isolation_level == "STRICT"

    async def test_isolation_statistics_tracking(self, sync_manager, sample_order):
        """测试隔离统计跟踪"""
        # 执行各种操作
        await sync_manager.register_order(sample_order, TradingMode.PAPER)
        await sync_manager.get_order(sample_order.client_order_id, TradingMode.PAPER)
        await sync_manager.get_order(sample_order.client_order_id)  # 跨环境查询
        
        # 尝试违规访问
        try:
            await sync_manager.update_order_status(
                sample_order.client_order_id,
                OrderStatus.FILLED,
                TradingMode.LIVE  # 错误的环境
            )
        except PermissionError:
            pass
        
        stats = sync_manager.isolation_stats
        assert stats["paper_operations"] >= 2
        assert stats["cross_environment_queries"] == 1
        assert stats["isolation_violations"] == 1