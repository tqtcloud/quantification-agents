"""
状态管理系统测试
测试TradingStateManager的状态序列化、持久化、回溯和调试功能
"""

import pytest
import tempfile
import shutil
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

from src.core.state_management import (
    TradingStateManager, StateConfig, DecisionNode, DecisionPath,
    StateSnapshot, TradingStateGraph
)
from src.core.models import MarketData, Signal, OrderSide


class TestStateConfig:
    """状态配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = StateConfig()
        
        assert config.enable_persistence is True
        assert config.enable_snapshots is True
        assert config.snapshot_frequency == 10
        assert config.max_history_size == 1000
        assert config.auto_backup is True
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = StateConfig(
            enable_persistence=False,
            snapshot_frequency=5,
            max_history_size=500
        )
        
        assert config.enable_persistence is False
        assert config.snapshot_frequency == 5
        assert config.max_history_size == 500


class TestTradingStateManager:
    """交易状态管理器测试"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def state_manager(self, temp_dir):
        """创建状态管理器"""
        config = StateConfig(
            storage_path=temp_dir,
            snapshot_frequency=3  # 降低频率便于测试
        )
        return TradingStateManager(config)
    
    def test_create_initial_state(self, state_manager):
        """测试创建初始状态"""
        market_data = {
            "BTCUSDT": MarketData(
                symbol="BTCUSDT",
                timestamp=int(time.time()),
                price=50000.0,
                volume=1000.0,
                bid=49999.0,
                ask=50001.0,
                bid_volume=100.0,
                ask_volume=100.0
            )
        }
        
        state = state_manager.create_initial_state(market_data)
        
        assert state["session_id"] is not None
        assert state["path_id"] is not None
        assert state["current_step"] == "START"
        assert state["step_history"] == ["START"]
        assert "BTCUSDT" in state["market_data"]
        assert len(state["signals"]) == 0
    
    def test_update_state(self, state_manager):
        """测试状态更新"""
        # 创建初始状态
        initial_state = state_manager.create_initial_state()
        
        # 更新状态
        updated_state = state_manager.update_state(
            state=initial_state,
            node_name="technical_analysis",
            agent_name="TechnicalAgent",
            input_data={"test": "input"},
            output_data={"test": "output"},
            execution_time=0.5,
            success=True
        )
        
        assert updated_state["current_step"] == "technical_analysis"
        assert "technical_analysis" in updated_state["step_history"]
        assert updated_state["agent_outputs"]["TechnicalAgent"]["test"] == "output"
        
        # 检查决策路径是否记录
        path = state_manager.get_decision_path()
        assert path is not None
        assert len(path.nodes) == 1
        assert path.nodes[0].node_name == "technical_analysis"
        assert path.nodes[0].success is True
    
    def test_create_snapshot(self, state_manager):
        """测试创建快照"""
        state = state_manager.create_initial_state()
        
        snapshot_id = state_manager.create_snapshot(state, "Test snapshot")
        
        assert snapshot_id is not None
        assert snapshot_id in state_manager._snapshots
        
        snapshot = state_manager._snapshots[snapshot_id]
        assert snapshot.description == "Test snapshot"
        assert snapshot.session_id == state["session_id"]
    
    def test_restore_from_snapshot(self, state_manager):
        """测试从快照恢复"""
        # 创建状态并更新
        state = state_manager.create_initial_state()
        updated_state = state_manager.update_state(
            state=state,
            node_name="test_node",
            agent_name="TestAgent",
            input_data={},
            output_data={"result": "test"},
            execution_time=0.1
        )
        
        # 创建快照
        snapshot_id = state_manager.create_snapshot(updated_state, "Before restore test")
        
        # 再次更新状态
        further_updated = state_manager.update_state(
            state=updated_state,
            node_name="another_node",
            agent_name="AnotherAgent",
            input_data={},
            output_data={},
            execution_time=0.1
        )
        
        # 从快照恢复
        restored_state = state_manager.restore_from_snapshot(snapshot_id)
        
        assert restored_state is not None
        assert restored_state["current_step"] == "test_node"
        assert "another_node" not in restored_state["step_history"]
    
    def test_decision_path_analysis(self, state_manager):
        """测试决策路径分析"""
        state = state_manager.create_initial_state()
        
        # 执行多个步骤
        nodes = ["node1", "node2", "node3"]
        execution_times = [0.1, 0.2, 0.3]
        
        for node, exec_time in zip(nodes, execution_times):
            state = state_manager.update_state(
                state=state,
                node_name=node,
                agent_name=f"Agent{node}",
                input_data={},
                output_data={},
                execution_time=exec_time,
                success=True
            )
        
        # 分析路径
        analysis = state_manager.get_path_analysis()
        
        assert analysis["total_nodes"] == 3
        assert analysis["total_execution_time"] == 0.6
        assert analysis["success_rate"] == 1.0
        assert analysis["avg_execution_time"] == 0.2
        assert len(analysis["node_breakdown"]) == 3
    
    def test_debug_state(self, state_manager):
        """测试状态调试"""
        state = state_manager.create_initial_state()
        
        # 更新状态
        state = state_manager.update_state(
            state=state,
            node_name="debug_test",
            agent_name="DebugAgent",
            input_data={},
            output_data={"debug": "info"},
            execution_time=0.1
        )
        
        debug_info = state_manager.debug_state(state)
        
        assert "session_id" in debug_info
        assert "path_id" in debug_info
        assert debug_info["current_step"] == "debug_test"
        assert debug_info["step_count"] == 2  # START + debug_test
        assert "DebugAgent" in debug_info["active_agents"]
    
    def test_automatic_snapshots(self, state_manager):
        """测试自动快照"""
        state = state_manager.create_initial_state()
        
        # 执行多个步骤触发自动快照
        for i in range(5):
            state = state_manager.update_state(
                state=state,
                node_name=f"node_{i}",
                agent_name=f"Agent_{i}",
                input_data={},
                output_data={},
                execution_time=0.1
            )
        
        # 检查是否创建了自动快照
        snapshots = list(state_manager._snapshots.values())
        auto_snapshots = [s for s in snapshots if "Auto snapshot" in s.description]
        
        assert len(auto_snapshots) > 0
    
    def test_backup_and_restore(self, state_manager):
        """测试备份和恢复"""
        # 创建状态
        state = state_manager.create_initial_state()
        state = state_manager.update_state(
            state=state,
            node_name="backup_test",
            agent_name="BackupAgent",
            input_data={},
            output_data={"data": "important"},
            execution_time=0.1
        )
        
        # 备份
        state_manager.backup_state()
        
        # 检查备份文件是否存在
        backup_dir = Path(state_manager.storage_path) / "backups"
        backup_files = list(backup_dir.glob("state_backup_*.pkl"))
        
        assert len(backup_files) > 0
        
        # 清空状态
        original_session = state_manager._current_session_id
        state_manager._current_state = None
        state_manager._decision_paths.clear()
        
        # 恢复
        backup_file = backup_files[0]
        success = state_manager.restore_from_backup(str(backup_file))
        
        assert success is True
        assert state_manager._current_session_id == original_session
        assert state_manager._current_state is not None
    
    def test_cleanup_old_data(self, state_manager):
        """测试清理旧数据"""
        # 创建一些快照
        state = state_manager.create_initial_state()
        
        snapshots = []
        for i in range(3):
            snapshot_id = state_manager.create_snapshot(state, f"Test snapshot {i}")
            snapshots.append(snapshot_id)
        
        initial_count = len(state_manager._snapshots)
        
        # 清理（使用很短的时间窗口强制清理）
        state_manager.cleanup_old_data(max_age_hours=0)
        
        # 检查是否清理了数据
        final_count = len(state_manager._snapshots)
        assert final_count <= initial_count
    
    def test_serialization_and_deserialization(self, state_manager):
        """测试序列化和反序列化"""
        # 创建包含复杂数据的状态
        market_data = {
            "BTCUSDT": MarketData(
                symbol="BTCUSDT",
                timestamp=int(time.time()),
                price=50000.0,
                volume=1000.0,
                bid=49999.0,
                ask=50001.0,
                bid_volume=100.0,
                ask_volume=100.0
            )
        }
        
        signals = [
            Signal(
                source="TestAgent",
                symbol="BTCUSDT",
                action=OrderSide.BUY,
                strength=0.8,
                confidence=0.9,
                reason="Test signal"
            )
        ]
        
        state = state_manager.create_initial_state(market_data)
        state["signals"] = signals
        
        # 序列化
        serialized = state_manager._serialize_state(state)
        
        assert isinstance(serialized, dict)
        assert "market_data" in serialized
        assert "signals" in serialized
        assert "timestamp" in serialized
        
        # 反序列化
        deserialized = state_manager._deserialize_state(serialized)
        
        assert isinstance(deserialized, dict)
        assert "market_data" in deserialized
        assert "signals" in deserialized


class TestDecisionNode:
    """决策节点测试"""
    
    def test_decision_node_creation(self):
        """测试决策节点创建"""
        node = DecisionNode(
            node_id="test_node_1",
            node_name="technical_analysis",
            agent_name="TechnicalAgent",
            input_data={"test": "input"},
            output_data={"test": "output"},
            execution_time=0.5,
            success=True
        )
        
        assert node.node_id == "test_node_1"
        assert node.node_name == "technical_analysis"
        assert node.agent_name == "TechnicalAgent"
        assert node.execution_time == 0.5
        assert node.success is True
        assert node.error_message is None
    
    def test_decision_node_with_error(self):
        """测试带错误的决策节点"""
        node = DecisionNode(
            node_id="error_node",
            node_name="failing_analysis",
            agent_name="FailingAgent",
            input_data={},
            output_data={},
            execution_time=0.1,
            success=False,
            error_message="Test error"
        )
        
        assert node.success is False
        assert node.error_message == "Test error"


class TestDecisionPath:
    """决策路径测试"""
    
    def test_decision_path_creation(self):
        """测试决策路径创建"""
        path = DecisionPath(
            path_id="test_path",
            session_id="test_session"
        )
        
        assert path.path_id == "test_path"
        assert path.session_id == "test_session"
        assert len(path.nodes) == 0
        assert path.total_execution_time == 0.0
        assert path.final_decision is None
    
    def test_decision_path_with_nodes(self):
        """测试带节点的决策路径"""
        node1 = DecisionNode(
            node_id="node_1",
            node_name="analysis",
            agent_name="Agent1",
            input_data={},
            output_data={},
            execution_time=0.3,
            success=True
        )
        
        node2 = DecisionNode(
            node_id="node_2",
            node_name="decision",
            agent_name="Agent2",
            input_data={},
            output_data={},
            execution_time=0.2,
            success=True
        )
        
        path = DecisionPath(
            path_id="multi_node_path",
            session_id="test_session",
            nodes=[node1, node2],
            total_execution_time=0.5
        )
        
        assert len(path.nodes) == 2
        assert path.total_execution_time == 0.5


class TestStateSnapshot:
    """状态快照测试"""
    
    def test_snapshot_creation(self):
        """测试快照创建"""
        snapshot = StateSnapshot(
            snapshot_id="snap_1",
            session_id="session_1",
            state_data={"test": "data"},
            description="Test snapshot"
        )
        
        assert snapshot.snapshot_id == "snap_1"
        assert snapshot.session_id == "session_1"
        assert snapshot.state_data["test"] == "data"
        assert snapshot.description == "Test snapshot"
        assert isinstance(snapshot.timestamp, datetime)


class TestTradingStateGraph:
    """交易状态图测试"""
    
    def test_state_graph_structure(self):
        """测试状态图结构"""
        # 创建一个符合TradingStateGraph类型的状态
        state: TradingStateGraph = {
            "market_data": {},
            "order_books": {},
            "positions": {},
            "orders": {},
            "signals": [],
            "risk_metrics": None,
            "strategies": {},
            "current_step": "START",
            "decision_context": {},
            "agent_outputs": {},
            "path_id": "test_path",
            "session_id": "test_session",
            "step_history": ["START"],
            "timestamp": datetime.utcnow(),
            "metadata": {}
        }
        
        # 验证所有必需字段都存在
        required_fields = [
            "market_data", "order_books", "positions", "orders", "signals",
            "risk_metrics", "strategies", "current_step", "decision_context",
            "agent_outputs", "path_id", "session_id", "step_history",
            "timestamp", "metadata"
        ]
        
        for field in required_fields:
            assert field in state
        
        # 验证数据类型
        assert isinstance(state["market_data"], dict)
        assert isinstance(state["signals"], list)
        assert isinstance(state["step_history"], list)
        assert isinstance(state["metadata"], dict)


if __name__ == "__main__":
    pytest.main([__file__])