"""
Agent状态管理器单元测试
测试状态管理、共享内存、序列化/反序列化、共识机制等功能
"""

import asyncio
import json
import pickle
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.models import (
    AgentState, AgentMessage, StateTransition, AgentConsensus,
    SharedMemory, AgentPerformanceState, StateCheckpoint,
    MarketDataState, RiskAssessmentState, PortfolioRecommendation,
    ReasoningStep, FinalDecision
)
from src.agents.state_manager import AgentStateManager, StateManagerConfig


@pytest.fixture
def temp_storage_path():
    """创建临时存储路径"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def state_manager_config(temp_storage_path):
    """创建测试配置"""
    return StateManagerConfig(
        enable_persistence=True,
        storage_path=temp_storage_path,
        max_state_history=10,
        state_ttl_seconds=3600,
        enable_compression=True,
        checkpoint_interval=60,
        enable_encryption=False,
        shared_memory_size_mb=10,
        consensus_timeout_seconds=5,
        enable_state_validation=True,
        auto_cleanup_interval=0  # 禁用自动清理以便测试
    )


@pytest.fixture
async def state_manager(state_manager_config):
    """创建状态管理器实例"""
    manager = AgentStateManager(state_manager_config)
    await manager.initialize()
    yield manager
    await manager.shutdown()


class TestAgentStateManager:
    """Agent状态管理器测试"""
    
    # ==================== 状态管理测试 ====================
    
    @pytest.mark.asyncio
    async def test_create_initial_state(self, state_manager):
        """测试创建初始状态"""
        session_id = "test_session"
        agent_id = "test_agent"
        
        state = state_manager.create_initial_state(session_id, agent_id)
        
        assert state["session_id"] == session_id
        assert state["agent_id"] == agent_id
        assert state["state_version"] == 1
        assert state["market_data"] == {}
        assert state["news_data"] == []
        assert state["social_sentiment"] == {}
        assert state["analyst_opinions"] == []
        assert state["confidence_scores"] == {}
        assert state["risk_assessment"]["risk_level"] == "medium"
        assert state["portfolio_recommendations"] == []
        assert state["final_decision"] is None
        assert state["reasoning_chain"] == []
        assert isinstance(state["created_at"], datetime)
        assert isinstance(state["updated_at"], datetime)
    
    @pytest.mark.asyncio
    async def test_get_state(self, state_manager):
        """测试获取状态"""
        session_id = "test_session"
        state_manager.create_initial_state(session_id)
        
        retrieved_state = await state_manager.get_state(session_id)
        
        assert retrieved_state is not None
        assert retrieved_state["session_id"] == session_id
        assert retrieved_state["state_version"] == 1
    
    @pytest.mark.asyncio
    async def test_update_state(self, state_manager):
        """测试更新状态"""
        session_id = "test_session"
        state_manager.create_initial_state(session_id)
        
        updates = {
            "confidence_scores": {"agent1": 0.8, "agent2": 0.9},
            "metadata": {"updated": True}
        }
        
        updated_state = await state_manager.update_state(
            session_id, updates, "test_agent"
        )
        
        assert updated_state["state_version"] == 2
        assert updated_state["confidence_scores"] == {"agent1": 0.8, "agent2": 0.9}
        assert updated_state["metadata"]["updated"] is True
        assert updated_state["agent_id"] == "test_agent"
    
    @pytest.mark.asyncio
    async def test_merge_states(self, state_manager):
        """测试合并状态"""
        session_id = "test_session"
        state_manager.create_initial_state(session_id)
        
        partial_state = {
            "market_data": {
                "BTC/USDT": {
                    "symbol": "BTC/USDT",
                    "price": 50000.0,
                    "volume": 1000.0,
                    "bid": 49999.0,
                    "ask": 50001.0,
                    "spread": 2.0,
                    "volatility": 0.02,
                    "trend": "up",
                    "momentum": 0.5,
                    "timestamp": datetime.now(),
                    "metadata": {}
                }
            },
            "confidence_scores": {"agent1": 0.7}
        }
        
        merged_state = await state_manager.merge_states(
            session_id, partial_state, "test_agent"
        )
        
        assert "BTC/USDT" in merged_state["market_data"]
        assert merged_state["market_data"]["BTC/USDT"]["price"] == 50000.0
        assert merged_state["confidence_scores"]["agent1"] == 0.7
        assert merged_state["state_version"] == 2
    
    @pytest.mark.asyncio
    async def test_state_history(self, state_manager):
        """测试状态历史"""
        session_id = "test_session"
        state_manager.create_initial_state(session_id)
        
        # 多次更新状态
        for i in range(5):
            await state_manager.update_state(
                session_id,
                {"metadata": {"iteration": i}},
                f"agent_{i}"
            )
        
        # 检查历史记录
        history = state_manager._state_history[session_id]
        assert len(history) == 6  # 初始状态 + 5次更新
        
        # 验证历史中的版本号
        versions = [s["state_version"] for s in history]
        assert versions == [1, 2, 3, 4, 5, 6]
    
    # ==================== 共享内存测试 ====================
    
    @pytest.mark.asyncio
    async def test_set_and_get_shared_memory(self, state_manager):
        """测试设置和获取共享内存"""
        key = "test_key"
        value = {"data": "test_value"}
        owner = "agent1"
        
        # 设置共享内存
        memory = await state_manager.set_shared_memory(key, value, owner, 3600)
        
        assert memory.key == key
        assert memory.value == value
        assert memory.owner_agent == owner
        assert memory.ttl_seconds == 3600
        
        # 获取共享内存
        retrieved_value = await state_manager.get_shared_memory(key, owner)
        assert retrieved_value == value
        
        # 其他agent也能访问
        retrieved_value2 = await state_manager.get_shared_memory(key, "agent2")
        assert retrieved_value2 == value
    
    @pytest.mark.asyncio
    async def test_shared_memory_access_control(self, state_manager):
        """测试共享内存访问控制"""
        key = "restricted_key"
        value = {"secret": "data"}
        owner = "owner_agent"
        
        # 设置有访问限制的共享内存
        memory = await state_manager.set_shared_memory(key, value, owner)
        memory.access_agents = ["agent1", "agent2"]
        state_manager._shared_memory[key] = memory
        
        # 允许的agent可以访问
        retrieved = await state_manager.get_shared_memory(key, "agent1")
        assert retrieved == value
        
        # 不允许的agent无法访问
        retrieved = await state_manager.get_shared_memory(key, "agent3")
        assert retrieved is None
        
        # 所有者始终可以访问
        retrieved = await state_manager.get_shared_memory(key, owner)
        assert retrieved == value
    
    @pytest.mark.asyncio
    async def test_shared_memory_locking(self, state_manager):
        """测试共享内存锁定"""
        key = "lock_test"
        value = {"data": "test"}
        owner = "owner_agent"
        
        await state_manager.set_shared_memory(key, value, owner)
        
        # agent1锁定内存
        locked = await state_manager.lock_shared_memory(key, "agent1")
        assert locked is True
        
        # agent2无法锁定
        locked = await state_manager.lock_shared_memory(key, "agent2")
        assert locked is False
        
        # agent1可以再次锁定
        locked = await state_manager.lock_shared_memory(key, "agent1")
        assert locked is True
        
        # agent1解锁
        unlocked = await state_manager.unlock_shared_memory(key, "agent1")
        assert unlocked is True
        
        # 现在agent2可以锁定
        locked = await state_manager.lock_shared_memory(key, "agent2")
        assert locked is True
    
    @pytest.mark.asyncio
    async def test_shared_memory_ttl(self, state_manager):
        """测试共享内存TTL"""
        key = "ttl_test"
        value = {"data": "expires"}
        owner = "agent1"
        
        # 设置TTL为0秒(立即过期)
        memory = await state_manager.set_shared_memory(key, value, owner, 0)
        
        # 修改创建时间为1小时前
        memory.created_at = datetime.now() - timedelta(hours=1)
        state_manager._shared_memory[key] = memory
        
        # 尝试获取应该返回None
        retrieved = await state_manager.get_shared_memory(key, owner)
        assert retrieved is None
        
        # 确认已被删除
        assert key not in state_manager._shared_memory
    
    # ==================== 消息传递测试 ====================
    
    @pytest.mark.asyncio
    async def test_send_and_receive_message(self, state_manager):
        """测试发送和接收消息"""
        message = AgentMessage(
            sender_agent="agent1",
            receiver_agent="agent2",
            message_type="test",
            payload={"data": "test_data"},
            priority=5
        )
        
        await state_manager.send_message(message)
        
        # 接收消息
        messages = await state_manager.receive_messages("agent2", 10)
        assert len(messages) == 1
        assert messages[0].sender_agent == "agent1"
        assert messages[0].payload["data"] == "test_data"
        
        # 再次接收应该为空
        messages = await state_manager.receive_messages("agent2", 10)
        assert len(messages) == 0
    
    @pytest.mark.asyncio
    async def test_broadcast_message(self, state_manager):
        """测试广播消息"""
        # 注册一些agent
        state_manager._message_handlers["agent1"] = []
        state_manager._message_handlers["agent2"] = []
        state_manager._message_handlers["agent3"] = []
        
        message = AgentMessage(
            sender_agent="broadcaster",
            receiver_agent=None,  # 广播
            message_type="broadcast",
            payload={"announcement": "test"}
        )
        
        await state_manager.send_message(message)
        
        # 所有agent都应该收到消息(除了发送者)
        for agent_id in ["agent1", "agent2", "agent3"]:
            messages = await state_manager.receive_messages(agent_id, 10)
            assert len(messages) == 1
            assert messages[0].payload["announcement"] == "test"
    
    @pytest.mark.asyncio
    async def test_message_handler(self, state_manager):
        """测试消息处理器"""
        received_messages = []
        
        async def handler(message: AgentMessage):
            received_messages.append(message)
        
        state_manager.register_message_handler("agent1", handler)
        
        message = AgentMessage(
            sender_agent="agent2",
            receiver_agent="agent1",
            message_type="test",
            payload={"data": "test"}
        )
        
        await state_manager.send_message(message)
        
        # 等待处理器执行
        await asyncio.sleep(0.1)
        
        assert len(received_messages) == 1
        assert received_messages[0].payload["data"] == "test"
    
    # ==================== 共识机制测试 ====================
    
    @pytest.mark.asyncio
    async def test_consensus_voting(self, state_manager):
        """测试共识投票"""
        agents = ["agent1", "agent2", "agent3"]
        consensus_id = await state_manager.initiate_consensus(
            "test_decision", agents, 0.6
        )
        
        # 提交投票
        await state_manager.submit_vote(consensus_id, "agent1", "approve", 0.9)
        await state_manager.submit_vote(consensus_id, "agent2", "approve", 0.8)
        await state_manager.submit_vote(consensus_id, "agent3", "reject", 0.7)
        
        # 获取结果
        result = await state_manager.get_consensus_result(consensus_id)
        
        assert result is not None
        assert result.consensus_reached is True
        assert result.final_decision == "approve"
        assert result.weighted_score > 0.6
    
    @pytest.mark.asyncio
    async def test_consensus_not_reached(self, state_manager):
        """测试未达成共识"""
        agents = ["agent1", "agent2", "agent3", "agent4"]
        consensus_id = await state_manager.initiate_consensus(
            "test_decision", agents, 0.8  # 高阈值
        )
        
        # 平均分配投票
        await state_manager.submit_vote(consensus_id, "agent1", "approve", 0.9)
        await state_manager.submit_vote(consensus_id, "agent2", "approve", 0.8)
        await state_manager.submit_vote(consensus_id, "agent3", "reject", 0.9)
        await state_manager.submit_vote(consensus_id, "agent4", "reject", 0.8)
        
        result = await state_manager.get_consensus_result(consensus_id)
        
        assert result is not None
        assert result.consensus_reached is False
        assert result.final_decision is None
        assert len(result.dissenting_opinions) > 0
    
    # ==================== 序列化测试 ====================
    
    def test_serialize_and_deserialize_state(self, state_manager):
        """测试状态序列化和反序列化"""
        # 创建一个复杂的状态
        state: AgentState = {
            "session_id": "test",
            "agent_id": "agent1",
            "state_version": 1,
            "market_data": {
                "BTC/USDT": {
                    "symbol": "BTC/USDT",
                    "price": 50000.0,
                    "volume": 1000.0,
                    "bid": 49999.0,
                    "ask": 50001.0,
                    "spread": 2.0,
                    "volatility": 0.02,
                    "trend": "up",
                    "momentum": 0.5,
                    "timestamp": datetime.now(),
                    "metadata": {"source": "exchange"}
                }
            },
            "news_data": [
                {
                    "source": "news_agency",
                    "title": "Market Update",
                    "content": "Bitcoin rises",
                    "sentiment_score": 0.8,
                    "relevance_score": 0.9,
                    "impact_level": "high",
                    "entities": ["Bitcoin"],
                    "timestamp": datetime.now(),
                    "url": "https://example.com"
                }
            ],
            "social_sentiment": {},
            "analyst_opinions": [],
            "confidence_scores": {"agent1": 0.8},
            "risk_assessment": {
                "risk_level": "low",
                "var_95": 0.05,
                "var_99": 0.08,
                "max_drawdown": 0.1,
                "sharpe_ratio": 1.5,
                "exposure_ratio": 0.6,
                "concentration_risk": 0.3,
                "liquidity_risk": 0.2,
                "market_risk": 0.4,
                "operational_risk": 0.1,
                "risk_factors": [],
                "mitigation_strategies": ["diversify"],
                "timestamp": datetime.now()
            },
            "portfolio_recommendations": [],
            "final_decision": None,
            "reasoning_chain": [],
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "metadata": {"test": True}
        }
        
        # 序列化
        serialized = state_manager.serialize_state(state)
        assert isinstance(serialized, bytes)
        
        # 反序列化
        deserialized = state_manager.deserialize_state(serialized)
        
        # 验证关键字段
        assert deserialized["session_id"] == "test"
        assert deserialized["agent_id"] == "agent1"
        assert deserialized["market_data"]["BTC/USDT"]["price"] == 50000.0
        assert len(deserialized["news_data"]) == 1
        assert deserialized["confidence_scores"]["agent1"] == 0.8
        assert deserialized["risk_assessment"]["risk_level"] == "low"
        assert isinstance(deserialized["created_at"], datetime)
    
    @pytest.mark.asyncio
    async def test_save_and_load_state(self, state_manager):
        """测试保存和加载状态"""
        session_id = "persist_test"
        state = state_manager.create_initial_state(session_id)
        
        # 更新状态
        await state_manager.update_state(
            session_id,
            {"metadata": {"saved": True}},
            "test_agent"
        )
        
        # 保存状态
        saved = await state_manager.save_state(session_id)
        assert saved is True
        
        # 清除内存中的状态
        del state_manager._current_states[session_id]
        
        # 加载状态
        loaded_state = await state_manager.load_state(session_id)
        
        assert loaded_state is not None
        assert loaded_state["session_id"] == session_id
        assert loaded_state["metadata"]["saved"] is True
        assert loaded_state["state_version"] == 2
    
    # ==================== 检查点测试 ====================
    
    @pytest.mark.asyncio
    async def test_create_and_restore_checkpoint(self, state_manager):
        """测试创建和恢复检查点"""
        session_id = "checkpoint_test"
        state = state_manager.create_initial_state(session_id)
        
        # 更新状态和性能数据
        await state_manager.update_state(
            session_id,
            {"metadata": {"checkpoint": True}},
            "test_agent"
        )
        
        state_manager.update_agent_performance(
            "test_agent", True, 0.9, 100.0, 1000.0
        )
        
        # 创建检查点
        checkpoint_id = await state_manager.create_checkpoint(
            session_id, "test", True
        )
        
        assert checkpoint_id is not None
        
        # 修改状态
        await state_manager.update_state(
            session_id,
            {"metadata": {"checkpoint": False, "modified": True}},
            "test_agent"
        )
        
        # 从检查点恢复
        restored = await state_manager.restore_from_checkpoint(checkpoint_id)
        assert restored is True
        
        # 验证恢复的状态
        current_state = await state_manager.get_state(session_id)
        assert current_state["metadata"]["checkpoint"] is True
        assert "modified" not in current_state["metadata"]
        
        # 验证恢复的性能数据
        performance = state_manager.get_agent_performance("test_agent")
        assert performance is not None
        assert performance.successful_decisions == 1
    
    # ==================== 性能追踪测试 ====================
    
    def test_update_agent_performance(self, state_manager):
        """测试更新Agent性能"""
        agent_name = "perf_agent"
        
        # 多次更新性能
        state_manager.update_agent_performance(agent_name, True, 0.9, 100.0, 500.0)
        state_manager.update_agent_performance(agent_name, True, 0.8, 150.0, 300.0)
        state_manager.update_agent_performance(agent_name, False, 0.6, 200.0, -200.0)
        
        performance = state_manager.get_agent_performance(agent_name)
        
        assert performance.total_decisions == 3
        assert performance.successful_decisions == 2
        assert performance.failed_decisions == 1
        assert performance.win_rate == pytest.approx(2/3)
        assert performance.average_confidence == pytest.approx((0.9 + 0.8 + 0.6) / 3)
        assert performance.average_processing_time_ms == pytest.approx((100 + 150 + 200) / 3)
        assert performance.total_profit_loss == 600.0
    
    # ==================== 统计分析测试 ====================
    
    @pytest.mark.asyncio
    async def test_get_state_statistics(self, state_manager):
        """测试获取状态统计"""
        session_id = "stats_test"
        state = state_manager.create_initial_state(session_id)
        
        # 添加一些数据
        partial_state = {
            "market_data": {"BTC/USDT": {}, "ETH/USDT": {}},
            "news_data": [{"title": "news1"}, {"title": "news2"}],
            "portfolio_recommendations": [{"symbol": "BTC"}],
            "reasoning_chain": [{"step_id": 1}, {"step_id": 2}],
            "confidence_scores": {"agent1": 0.8, "agent2": 0.9}
        }
        
        await state_manager.merge_states(session_id, partial_state, "test_agent")
        
        stats = state_manager.get_state_statistics(session_id)
        
        assert stats["session_id"] == session_id
        assert stats["state_version"] == 2
        assert stats["market_data_symbols"] == 2
        assert stats["news_count"] == 2
        assert stats["recommendations_count"] == 1
        assert stats["reasoning_steps"] == 2
        assert stats["active_agents"] == 2
        assert stats["has_final_decision"] is False
    
    @pytest.mark.asyncio
    async def test_get_reasoning_summary(self, state_manager):
        """测试获取推理摘要"""
        session_id = "reasoning_test"
        state = state_manager.create_initial_state(session_id)
        
        # 添加推理链
        reasoning_chain = [
            {
                "step_id": 1,
                "agent_name": "agent1",
                "action": "analyze",
                "input_data": {},
                "output_data": {},
                "confidence": 0.8,
                "reasoning": "Initial analysis",
                "timestamp": datetime.now(),
                "duration_ms": 100.0
            },
            {
                "step_id": 2,
                "agent_name": "agent2",
                "action": "validate",
                "input_data": {},
                "output_data": {},
                "confidence": 0.9,
                "reasoning": "Validation complete",
                "timestamp": datetime.now(),
                "duration_ms": 50.0
            }
        ]
        
        await state_manager.update_state(
            session_id,
            {"reasoning_chain": reasoning_chain},
            "test_agent"
        )
        
        summary = state_manager.get_reasoning_summary(session_id)
        
        assert len(summary) == 2
        assert summary[0]["step_id"] == 1
        assert summary[0]["agent_name"] == "agent1"
        assert summary[0]["confidence"] == 0.8
        assert summary[1]["step_id"] == 2
        assert summary[1]["agent_name"] == "agent2"
        assert summary[1]["confidence"] == 0.9
    
    # ==================== 数据质量测试 ====================
    
    def test_assess_data_quality(self, state_manager):
        """测试数据质量评估"""
        # 完整数据
        complete_data = {
            "price": 50000.0,
            "volume": 1000.0,
            "timestamp": datetime.now()
        }
        
        quality = state_manager._assess_data_quality(complete_data)
        assert quality.completeness_score == 1.0
        assert quality.accuracy_score == 1.0
        assert quality.overall_quality == 1.0
        assert len(quality.missing_fields) == 0
        assert len(quality.anomalies_detected) == 0
        
        # 不完整数据
        incomplete_data = {
            "price": None,
            "volume": 1000.0
        }
        
        quality = state_manager._assess_data_quality(incomplete_data)
        assert quality.completeness_score < 1.0
        assert "price" in quality.missing_fields
        assert "timestamp" in quality.missing_fields
        
        # 异常数据
        anomaly_data = {
            "price": -100.0,  # 负价格
            "volume": 1000.0,
            "timestamp": datetime.now()
        }
        
        quality = state_manager._assess_data_quality(anomaly_data)
        assert quality.accuracy_score < 1.0
        assert "negative_or_zero_price" in quality.anomalies_detected
    
    # ==================== 清理测试 ====================
    
    @pytest.mark.asyncio
    async def test_cleanup_expired_data(self, state_manager):
        """测试清理过期数据"""
        # 添加过期的共享内存
        memory1 = SharedMemory(
            memory_id="mem1",
            key="expired",
            value="data",
            owner_agent="agent1",
            ttl_seconds=1
        )
        memory1.created_at = datetime.now() - timedelta(hours=2)
        state_manager._shared_memory["expired"] = memory1
        
        # 添加未过期的共享内存
        memory2 = SharedMemory(
            memory_id="mem2",
            key="valid",
            value="data",
            owner_agent="agent2",
            ttl_seconds=3600
        )
        state_manager._shared_memory["valid"] = memory2
        
        # 执行清理
        await state_manager._cleanup_expired_data()
        
        # 验证过期数据被清理
        assert "expired" not in state_manager._shared_memory
        assert "valid" in state_manager._shared_memory
    
    # ==================== 边缘情况测试 ====================
    
    @pytest.mark.asyncio
    async def test_update_nonexistent_state(self, state_manager):
        """测试更新不存在的状态"""
        with pytest.raises(ValueError, match="Session .* not found"):
            await state_manager.update_state(
                "nonexistent", {"data": "test"}, "agent1"
            )
    
    @pytest.mark.asyncio
    async def test_invalid_consensus_vote(self, state_manager):
        """测试无效的共识投票"""
        # 不存在的共识
        with pytest.raises(ValueError, match="Consensus .* not found"):
            await state_manager.submit_vote("invalid_id", "agent1", "vote", 0.5)
        
        # 未参与的agent投票
        consensus_id = await state_manager.initiate_consensus(
            "test", ["agent1", "agent2"], 0.5
        )
        
        with pytest.raises(ValueError, match="Agent .* not participating"):
            await state_manager.submit_vote(consensus_id, "agent3", "vote", 0.5)
    
    @pytest.mark.asyncio
    async def test_concurrent_state_updates(self, state_manager):
        """测试并发状态更新"""
        session_id = "concurrent_test"
        state_manager.create_initial_state(session_id)
        
        async def update_task(agent_id: str, value: int):
            await state_manager.update_state(
                session_id,
                {"metadata": {agent_id: value}},
                agent_id
            )
        
        # 并发执行多个更新
        tasks = [
            update_task(f"agent_{i}", i)
            for i in range(10)
        ]
        
        await asyncio.gather(*tasks)
        
        # 验证所有更新都成功
        final_state = await state_manager.get_state(session_id)
        
        for i in range(10):
            assert final_state["metadata"][f"agent_{i}"] == i
        
        # 状态版本应该递增了10次
        assert final_state["state_version"] == 11  # 初始1 + 10次更新