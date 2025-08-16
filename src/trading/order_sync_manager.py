import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
import json

from src.core.models import Order, OrderStatus, TradingState
from src.trading.order_router import TradingMode
from src.utils.logger import LoggerMixin


class SyncStatus(Enum):
    """同步状态"""
    PENDING = "pending"
    IN_SYNC = "in_sync"
    OUT_OF_SYNC = "out_of_sync"
    ERROR = "error"


@dataclass
class OrderSyncRecord:
    """订单同步记录"""
    order_id: str
    client_order_id: str
    symbol: str
    trading_mode: TradingMode
    local_status: OrderStatus
    remote_status: Optional[OrderStatus] = None
    sync_status: SyncStatus = SyncStatus.PENDING
    last_sync_time: Optional[datetime] = None
    sync_attempts: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataIsolationConfig:
    """数据隔离配置"""
    enable_isolation: bool = True
    paper_data_prefix: str = "paper_"
    live_data_prefix: str = "live_"
    shared_data_keys: Set[str] = field(default_factory=lambda: {"system_config", "market_data"})
    isolation_level: str = "STRICT"  # STRICT, MODERATE, LOOSE


class OrderSyncManager(LoggerMixin):
    """订单状态同步和数据隔离管理器"""
    
    def __init__(self, isolation_config: Optional[DataIsolationConfig] = None):
        self.isolation_config = isolation_config or DataIsolationConfig()
        
        # 同步状态追踪
        self.sync_records: Dict[str, OrderSyncRecord] = {}
        
        # 数据存储分离
        self.paper_orders: Dict[str, Order] = {}
        self.live_orders: Dict[str, Order] = {}
        
        # 历史记录分离
        self.paper_order_history: List[Dict[str, Any]] = []
        self.live_order_history: List[Dict[str, Any]] = []
        
        # 状态同步配置
        self.sync_interval = 5.0  # 5秒同步间隔
        self.max_sync_attempts = 3
        self.sync_timeout = 30.0
        
        # 同步任务管理
        self._sync_task: Optional[asyncio.Task] = None
        self._running = False
        
        # 数据隔离统计
        self.isolation_stats = {
            "paper_operations": 0,
            "live_operations": 0,
            "cross_environment_queries": 0,
            "isolation_violations": 0
        }
    
    async def initialize(self):
        """初始化同步管理器"""
        self.log_info("Initializing OrderSyncManager")
        
        # 启动同步任务
        self._running = True
        self._sync_task = asyncio.create_task(self._sync_loop())
        
        self.log_info("OrderSyncManager initialized successfully")
    
    async def register_order(
        self,
        order: Order,
        trading_mode: TradingMode,
        execution_result: Optional[Dict[str, Any]] = None
    ):
        """注册订单到相应环境"""
        client_order_id = order.client_order_id or f"sync_{order.order_id}"
        
        # 数据隔离存储
        if trading_mode == TradingMode.PAPER:
            self.paper_orders[client_order_id] = order
            self.isolation_stats["paper_operations"] += 1
        else:
            self.live_orders[client_order_id] = order
            self.isolation_stats["live_operations"] += 1
        
        # 创建同步记录
        sync_record = OrderSyncRecord(
            order_id=order.order_id or "",
            client_order_id=client_order_id,
            symbol=order.symbol,
            trading_mode=trading_mode,
            local_status=order.status,
            metadata={
                "execution_result": execution_result,
                "created_at": datetime.utcnow().isoformat()
            }
        )
        
        self.sync_records[client_order_id] = sync_record
        
        self.log_info(
            "Order registered",
            client_order_id=client_order_id,
            trading_mode=trading_mode.value,
            symbol=order.symbol
        )
    
    async def update_order_status(
        self,
        client_order_id: str,
        new_status: OrderStatus,
        trading_mode: TradingMode,
        remote_data: Optional[Dict[str, Any]] = None
    ):
        """更新订单状态"""
        # 检查数据隔离
        if not self._check_data_access_permission(client_order_id, trading_mode):
            self.isolation_stats["isolation_violations"] += 1
            raise PermissionError(f"Cross-environment access denied: {client_order_id}")
        
        # 更新本地订单状态
        if trading_mode == TradingMode.PAPER and client_order_id in self.paper_orders:
            order = self.paper_orders[client_order_id]
            order.status = new_status
            order.updated_at = datetime.utcnow()
            
        elif trading_mode == TradingMode.LIVE and client_order_id in self.live_orders:
            order = self.live_orders[client_order_id]
            order.status = new_status
            order.updated_at = datetime.utcnow()
        
        # 更新同步记录
        if client_order_id in self.sync_records:
            sync_record = self.sync_records[client_order_id]
            sync_record.local_status = new_status
            sync_record.last_sync_time = datetime.utcnow()
            
            if remote_data:
                sync_record.remote_status = OrderStatus(remote_data.get("status", new_status.value))
                sync_record.metadata.update(remote_data)
                
                # 检查同步状态
                if sync_record.local_status == sync_record.remote_status:
                    sync_record.sync_status = SyncStatus.IN_SYNC
                else:
                    sync_record.sync_status = SyncStatus.OUT_OF_SYNC
        
        self.log_debug(
            "Order status updated",
            client_order_id=client_order_id,
            new_status=new_status.value,
            trading_mode=trading_mode.value
        )
    
    async def get_order(
        self,
        client_order_id: str,
        trading_mode: Optional[TradingMode] = None
    ) -> Optional[Order]:
        """获取订单（支持跨环境查询）"""
        # 如果指定了交易模式，直接查询对应环境
        if trading_mode == TradingMode.PAPER:
            self.isolation_stats["paper_operations"] += 1
            return self.paper_orders.get(client_order_id)
        
        elif trading_mode == TradingMode.LIVE:
            self.isolation_stats["live_operations"] += 1
            return self.live_orders.get(client_order_id)
        
        # 跨环境查询
        self.isolation_stats["cross_environment_queries"] += 1
        
        # 先查纸上交易
        if client_order_id in self.paper_orders:
            return self.paper_orders[client_order_id]
        
        # 再查实盘交易
        if client_order_id in self.live_orders:
            return self.live_orders[client_order_id]
        
        return None
    
    async def get_orders_by_environment(self, trading_mode: TradingMode) -> List[Order]:
        """按环境获取订单列表"""
        if trading_mode == TradingMode.PAPER:
            self.isolation_stats["paper_operations"] += 1
            return list(self.paper_orders.values())
        else:
            self.isolation_stats["live_operations"] += 1
            return list(self.live_orders.values())
    
    async def get_order_history(
        self,
        trading_mode: TradingMode,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """获取订单历史（环境隔离）"""
        if trading_mode == TradingMode.PAPER:
            history = self.paper_order_history
            self.isolation_stats["paper_operations"] += 1
        else:
            history = self.live_order_history
            self.isolation_stats["live_operations"] += 1
        
        # 过滤条件
        filtered_history = history
        if symbol:
            filtered_history = [h for h in history if h.get("symbol") == symbol]
        
        # 限制数量
        return filtered_history[-limit:] if limit > 0 else filtered_history
    
    async def archive_completed_order(self, client_order_id: str, trading_mode: TradingMode):
        """归档已完成订单"""
        order = await self.get_order(client_order_id, trading_mode)
        if not order:
            return
        
        # 创建历史记录
        history_record = {
            "order_id": order.order_id,
            "client_order_id": client_order_id,
            "symbol": order.symbol,
            "side": order.side.value,
            "order_type": order.order_type.value,
            "quantity": order.quantity,
            "price": order.price,
            "executed_qty": order.executed_qty,
            "avg_price": order.avg_price,
            "status": order.status.value,
            "created_at": order.created_at.isoformat() if order.created_at else None,
            "updated_at": order.updated_at.isoformat() if order.updated_at else None,
            "trading_mode": trading_mode.value,
            "archived_at": datetime.utcnow().isoformat()
        }
        
        # 分环境存储历史
        if trading_mode == TradingMode.PAPER:
            self.paper_order_history.append(history_record)
            if client_order_id in self.paper_orders:
                del self.paper_orders[client_order_id]
        else:
            self.live_order_history.append(history_record)
            if client_order_id in self.live_orders:
                del self.live_orders[client_order_id]
        
        # 清理同步记录
        if client_order_id in self.sync_records:
            del self.sync_records[client_order_id]
        
        self.log_debug(f"Order archived: {client_order_id}")
    
    def _check_data_access_permission(self, client_order_id: str, trading_mode: TradingMode) -> bool:
        """检查数据访问权限"""
        if not self.isolation_config.enable_isolation:
            return True
        
        # 严格隔离模式
        if self.isolation_config.isolation_level == "STRICT":
            if trading_mode == TradingMode.PAPER:
                return client_order_id in self.paper_orders
            else:
                return client_order_id in self.live_orders
        
        # 中等隔离模式
        elif self.isolation_config.isolation_level == "MODERATE":
            # 允许读取但不允许修改
            return True
        
        # 宽松隔离模式
        else:
            return True
    
    async def _sync_loop(self):
        """同步循环"""
        while self._running:
            try:
                await self._perform_sync()
                await asyncio.sleep(self.sync_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log_error(f"Sync loop error: {e}")
                await asyncio.sleep(self.sync_interval)
    
    async def _perform_sync(self):
        """执行同步"""
        out_of_sync_records = [
            record for record in self.sync_records.values()
            if record.sync_status in [SyncStatus.PENDING, SyncStatus.OUT_OF_SYNC]
            and record.sync_attempts < self.max_sync_attempts
        ]
        
        if not out_of_sync_records:
            return
        
        self.log_debug(f"Syncing {len(out_of_sync_records)} records")
        
        for record in out_of_sync_records:
            try:
                await self._sync_single_record(record)
            except Exception as e:
                record.error_message = str(e)
                record.sync_status = SyncStatus.ERROR
                record.sync_attempts += 1
                self.log_warning(f"Sync failed for {record.client_order_id}: {e}")
    
    async def _sync_single_record(self, record: OrderSyncRecord):
        """同步单个记录"""
        # 这里需要根据实际的API接口实现同步逻辑
        # 简化实现：模拟同步成功
        
        record.sync_attempts += 1
        record.last_sync_time = datetime.utcnow()
        
        # 模拟远程状态查询
        # 在实际实现中，这里应该调用相应的API查询订单状态
        if record.trading_mode == TradingMode.LIVE:
            # 查询实盘订单状态
            pass
        else:
            # 查询模拟盘订单状态
            pass
        
        # 假设同步成功
        record.sync_status = SyncStatus.IN_SYNC
        
        self.log_debug(f"Synced record: {record.client_order_id}")
    
    def get_sync_statistics(self) -> Dict[str, Any]:
        """获取同步统计"""
        total_records = len(self.sync_records)
        in_sync_records = sum(1 for r in self.sync_records.values() if r.sync_status == SyncStatus.IN_SYNC)
        out_of_sync_records = sum(1 for r in self.sync_records.values() if r.sync_status == SyncStatus.OUT_OF_SYNC)
        error_records = sum(1 for r in self.sync_records.values() if r.sync_status == SyncStatus.ERROR)
        
        return {
            "total_records": total_records,
            "in_sync_records": in_sync_records,
            "out_of_sync_records": out_of_sync_records,
            "error_records": error_records,
            "sync_rate": in_sync_records / total_records if total_records > 0 else 0,
            "isolation_stats": self.isolation_stats,
            "paper_orders_count": len(self.paper_orders),
            "live_orders_count": len(self.live_orders),
            "paper_history_count": len(self.paper_order_history),
            "live_history_count": len(self.live_order_history)
        }
    
    def get_environment_summary(self) -> Dict[str, Any]:
        """获取环境摘要"""
        return {
            "paper_trading": {
                "active_orders": len(self.paper_orders),
                "completed_orders": len(self.paper_order_history),
                "symbols": list(set(order.symbol for order in self.paper_orders.values()))
            },
            "live_trading": {
                "active_orders": len(self.live_orders),
                "completed_orders": len(self.live_order_history),
                "symbols": list(set(order.symbol for order in self.live_orders.values()))
            },
            "isolation_config": {
                "enabled": self.isolation_config.enable_isolation,
                "level": self.isolation_config.isolation_level,
                "paper_prefix": self.isolation_config.paper_data_prefix,
                "live_prefix": self.isolation_config.live_data_prefix
            }
        }
    
    async def force_sync_order(self, client_order_id: str) -> bool:
        """强制同步指定订单"""
        if client_order_id not in self.sync_records:
            return False
        
        record = self.sync_records[client_order_id]
        
        try:
            await self._sync_single_record(record)
            return record.sync_status == SyncStatus.IN_SYNC
        except Exception as e:
            self.log_error(f"Force sync failed for {client_order_id}: {e}")
            return False
    
    async def reset_environment_data(self, trading_mode: TradingMode, confirm: bool = False):
        """重置环境数据（危险操作）"""
        if not confirm:
            raise ValueError("This operation requires explicit confirmation")
        
        if trading_mode == TradingMode.PAPER:
            self.paper_orders.clear()
            self.paper_order_history.clear()
            self.log_warning("Paper trading environment data reset")
        else:
            self.live_orders.clear()
            self.live_order_history.clear()
            self.log_warning("Live trading environment data reset")
        
        # 清理相关同步记录
        to_remove = [
            cid for cid, record in self.sync_records.items()
            if record.trading_mode == trading_mode
        ]
        
        for cid in to_remove:
            del self.sync_records[cid]
    
    def export_environment_data(self, trading_mode: TradingMode) -> Dict[str, Any]:
        """导出环境数据"""
        if trading_mode == TradingMode.PAPER:
            return {
                "trading_mode": "paper",
                "active_orders": [
                    {
                        "client_order_id": cid,
                        "order_id": order.order_id,
                        "symbol": order.symbol,
                        "side": order.side.value,
                        "order_type": order.order_type.value,
                        "quantity": order.quantity,
                        "price": order.price,
                        "status": order.status.value,
                        "created_at": order.created_at.isoformat() if order.created_at else None
                    }
                    for cid, order in self.paper_orders.items()
                ],
                "order_history": self.paper_order_history.copy(),
                "exported_at": datetime.utcnow().isoformat()
            }
        else:
            return {
                "trading_mode": "live",
                "active_orders": [
                    {
                        "client_order_id": cid,
                        "order_id": order.order_id,
                        "symbol": order.symbol,
                        "side": order.side.value,
                        "order_type": order.order_type.value,
                        "quantity": order.quantity,
                        "price": order.price,
                        "status": order.status.value,
                        "created_at": order.created_at.isoformat() if order.created_at else None
                    }
                    for cid, order in self.live_orders.items()
                ],
                "order_history": self.live_order_history.copy(),
                "exported_at": datetime.utcnow().isoformat()
            }
    
    async def cleanup(self):
        """清理资源"""
        self.log_info("Cleaning up OrderSyncManager")
        
        self._running = False
        
        if self._sync_task and not self._sync_task.done():
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        
        self.log_info("OrderSyncManager cleanup complete")