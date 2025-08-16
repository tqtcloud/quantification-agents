"""
消息类型和路由测试
测试标准消息格式、路由器、主题构建器等功能
"""

import pytest
import time
from unittest.mock import MagicMock

from src.core.message_types import (
    MessageType, SignalType, RiskLevel,
    MarketTickMessage, MarketOrderBookMessage, MarketKlineMessage,
    TechnicalSignalMessage, SentimentSignalMessage, OrderMessage,
    PositionMessage, RiskAlertMessage, SystemMessage, AgentMessage,
    MessageRouter, TopicBuilder
)
from src.core.models import OrderSide, OrderStatus, OrderType


class TestMessageEnums:
    """消息类型枚举测试"""
    
    def test_message_type_enum(self):
        """测试消息类型枚举"""
        assert MessageType.MARKET_TICK.value == "market.tick"
        assert MessageType.ORDER_NEW.value == "order.new"
        assert MessageType.RISK_ALERT.value == "risk.alert"
        assert MessageType.SYSTEM_ERROR.value == "system.error"
        assert MessageType.AGENT_REGISTER.value == "agent.register"
    
    def test_signal_type_enum(self):
        """测试信号类型枚举"""
        assert SignalType.BUY.value == "buy"
        assert SignalType.SELL.value == "sell"
        assert SignalType.HOLD.value == "hold"
        assert SignalType.EXIT.value == "exit"
    
    def test_risk_level_enum(self):
        """测试风险级别枚举"""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.CRITICAL.value == "critical"


class TestMarketMessages:
    """市场数据消息测试"""
    
    def test_market_tick_message(self):
        """测试市场Tick消息"""
        message = MarketTickMessage(
            symbol="BTCUSDT",
            price=50000.0,
            volume=1.5,
            bid_price=49999.0,
            ask_price=50001.0,
            bid_volume=10.0,
            ask_volume=12.0,
            exchange="binance"
        )
        
        assert message.symbol == "BTCUSDT"
        assert message.price == 50000.0
        assert message.to_topic() == "market.btcusdt.tick"
        assert message.exchange == "binance"
        assert isinstance(message.timestamp, float)
    
    def test_market_orderbook_message(self):
        """测试订单簿消息"""
        message = MarketOrderBookMessage(
            symbol="ETHUSDT",
            bids=[[3000.0, 5.0], [2999.0, 10.0]],
            asks=[[3001.0, 8.0], [3002.0, 15.0]],
            last_update_id=12345
        )
        
        assert message.symbol == "ETHUSDT"
        assert len(message.bids) == 2
        assert len(message.asks) == 2
        assert message.to_topic() == "market.ethusdt.orderbook"
        assert message.last_update_id == 12345
    
    def test_market_kline_message(self):
        """测试K线消息"""
        current_time = int(time.time() * 1000)
        
        message = MarketKlineMessage(
            symbol="ADAUSDT",
            interval="1m",
            open_price=1.20,
            high_price=1.25,
            low_price=1.18,
            close_price=1.22,
            volume=1000.0,
            open_time=current_time,
            close_time=current_time + 60000
        )
        
        assert message.symbol == "ADAUSDT"
        assert message.interval == "1m"
        assert message.to_topic() == "market.adausdt.kline.1m"
        assert message.open_price == 1.20
        assert message.close_price == 1.22


class TestSignalMessages:
    """信号消息测试"""
    
    def test_technical_signal_message(self):
        """测试技术信号消息"""
        message = TechnicalSignalMessage(
            symbol="BTCUSDT",
            indicator="RSI",
            signal_type=SignalType.BUY,
            strength=0.8,
            confidence=0.9,
            price=50000.0,
            value=30.0,
            timeframe="5m",
            metadata={"period": 14, "source": "close"}
        )
        
        assert message.symbol == "BTCUSDT"
        assert message.indicator == "RSI"
        assert message.signal_type == SignalType.BUY
        assert message.to_topic() == "signal.technical.rsi.btcusdt"
        assert message.metadata["period"] == 14
    
    def test_sentiment_signal_message(self):
        """测试情绪信号消息"""
        message = SentimentSignalMessage(
            symbol="ETHUSDT",
            source="twitter",
            sentiment_score=0.7,
            confidence=0.85,
            signal_type=SignalType.BUY,
            content="Ethereum is looking bullish!",
            metadata={"tweet_count": 150}
        )
        
        assert message.symbol == "ETHUSDT"
        assert message.source == "twitter"
        assert message.sentiment_score == 0.7
        assert message.to_topic() == "signal.sentiment.twitter.ethusdt"
        assert message.content == "Ethereum is looking bullish!"


class TestTradingMessages:
    """交易相关消息测试"""
    
    def test_order_message(self):
        """测试订单消息"""
        message = OrderMessage(
            order_id="order_123",
            client_order_id="client_456",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.5,
            price=49000.0,
            status=OrderStatus.NEW,
            filled_quantity=0.0,
            metadata={"strategy": "momentum"}
        )
        
        assert message.order_id == "order_123"
        assert message.symbol == "BTCUSDT"
        assert message.side == OrderSide.BUY
        assert message.to_topic() == "order.new.btcusdt"
        assert message.metadata["strategy"] == "momentum"
    
    def test_position_message(self):
        """测试仓位消息"""
        message = PositionMessage(
            symbol="ETHUSDT",
            side=OrderSide.BUY,
            quantity=2.0,
            entry_price=3000.0,
            current_price=3100.0,
            unrealized_pnl=200.0,
            realized_pnl=50.0,
            margin=300.0
        )
        
        assert message.symbol == "ETHUSDT"
        assert message.side == OrderSide.BUY
        assert message.quantity == 2.0
        assert message.to_topic() == "position.update.ethusdt"
        assert message.unrealized_pnl == 200.0


class TestSystemMessages:
    """系统消息测试"""
    
    def test_risk_alert_message(self):
        """测试风险警告消息"""
        message = RiskAlertMessage(
            risk_type="drawdown",
            level=RiskLevel.HIGH,
            symbol="BTCUSDT",
            current_value=0.15,
            threshold=0.10,
            description="Portfolio drawdown exceeded threshold",
            metadata={"portfolio_value": 10000}
        )
        
        assert message.risk_type == "drawdown"
        assert message.level == RiskLevel.HIGH
        assert message.to_topic() == "risk.alert.high.btcusdt"
        assert message.current_value > message.threshold
    
    def test_risk_alert_message_no_symbol(self):
        """测试无特定交易对的风险警告消息"""
        message = RiskAlertMessage(
            risk_type="exposure",
            level=RiskLevel.CRITICAL,
            symbol=None,
            current_value=0.8,
            threshold=0.7,
            description="Overall exposure too high"
        )
        
        assert message.symbol is None
        assert message.to_topic() == "risk.alert.critical"
    
    def test_system_message(self):
        """测试系统消息"""
        message = SystemMessage(
            event_type="start",
            component="trading_engine",
            status="active",
            message="Trading engine started successfully",
            metadata={"version": "1.0.0"}
        )
        
        assert message.event_type == "start"
        assert message.component == "trading_engine"
        assert message.to_topic() == "system.start.trading_engine"
        assert message.metadata["version"] == "1.0.0"
    
    def test_agent_message(self):
        """测试Agent消息"""
        message = AgentMessage(
            agent_id="agent_001",
            event_type="register",
            agent_type="technical",
            status="active",
            config={"indicators": ["RSI", "MACD"]},
            metadata={"startup_time": time.time()}
        )
        
        assert message.agent_id == "agent_001"
        assert message.agent_type == "technical"
        assert message.to_topic() == "agent.register.technical"
        assert "RSI" in message.config["indicators"]


class TestMessageRouter:
    """消息路由器测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.router = MessageRouter()
    
    def test_add_route(self):
        """测试添加路由规则"""
        self.router.add_route("market.*", ["storage.market", "analysis.technical"])
        
        assert "market.*" in self.router.routing_rules
        assert self.router.routing_rules["market.*"] == ["storage.market", "analysis.technical"]
    
    def test_add_filter(self):
        """测试添加过滤器"""
        def btc_filter(message):
            return hasattr(message, 'symbol') and 'BTC' in message.symbol
        
        self.router.add_filter("market.*", btc_filter)
        
        assert "market.*" in self.router.filters
        assert self.router.filters["market.*"] == btc_filter
    
    def test_route_message(self):
        """测试消息路由"""
        # 添加路由规则
        self.router.add_route("market.*", ["storage.market", "analysis.technical"])
        self.router.add_route("signal.*", ["decision.maker"])
        
        # 测试市场消息路由
        market_message = MarketTickMessage(
            symbol="BTCUSDT",
            price=50000.0,
            volume=1.0,
            bid_price=49999.0,
            ask_price=50001.0,
            bid_volume=5.0,
            ask_volume=5.0
        )
        
        targets = self.router.route_message(market_message)
        assert "storage.market" in targets
        assert "analysis.technical" in targets
        
        # 测试信号消息路由
        signal_message = TechnicalSignalMessage(
            symbol="ETHUSDT",
            indicator="MACD",
            signal_type=SignalType.BUY,
            strength=0.8,
            confidence=0.9,
            price=3000.0,
            value=100.0
        )
        
        targets = self.router.route_message(signal_message)
        assert "decision.maker" in targets
    
    def test_route_message_with_filter(self):
        """测试带过滤器的消息路由"""
        # 添加路由规则
        self.router.add_route("signal.*", ["high_priority", "low_priority"])
        
        # 添加过滤器（只有高强度信号才路由到高优先级）
        def high_strength_filter(message):
            return hasattr(message, 'strength') and message.strength > 0.8
        
        self.router.add_filter("high_priority", high_strength_filter)
        
        # 测试高强度信号
        high_signal = TechnicalSignalMessage(
            symbol="BTCUSDT",
            indicator="RSI",
            signal_type=SignalType.BUY,
            strength=0.9,
            confidence=0.8,
            price=50000.0,
            value=25.0
        )
        
        targets = self.router.route_message(high_signal)
        assert "high_priority" in targets
        assert "low_priority" in targets
        
        # 测试低强度信号
        low_signal = TechnicalSignalMessage(
            symbol="BTCUSDT",
            indicator="RSI",
            signal_type=SignalType.BUY,
            strength=0.5,
            confidence=0.8,
            price=50000.0,
            value=25.0
        )
        
        targets = self.router.route_message(low_signal)
        assert "high_priority" not in targets
        assert "low_priority" in targets
    
    def test_pattern_matching(self):
        """测试模式匹配"""
        # 测试通配符匹配
        assert self.router._match_pattern("market.btcusdt.tick", "market.*")
        assert self.router._match_pattern("signal.technical.rsi", "signal.*")
        assert not self.router._match_pattern("order.new", "market.*")
        
        # 测试前缀匹配
        assert self.router._match_pattern("market.btcusdt.tick", "market*")
        
        # 测试后缀匹配
        assert self.router._match_pattern("market.btcusdt.tick", "*tick")
        
        # 测试精确匹配
        assert self.router._match_pattern("exact.match", "exact.match")
        assert not self.router._match_pattern("exact.match", "different.topic")


class TestTopicBuilder:
    """主题构建器测试"""
    
    def test_market_topics(self):
        """测试市场数据主题构建"""
        assert TopicBuilder.market_tick("BTCUSDT") == "market.btcusdt.tick"
        assert TopicBuilder.market_orderbook("ETHUSDT") == "market.ethusdt.orderbook"
        assert TopicBuilder.market_kline("ADAUSDT", "5m") == "market.adausdt.kline.5m"
    
    def test_signal_topics(self):
        """测试信号主题构建"""
        assert TopicBuilder.technical_signal("RSI", "BTCUSDT") == "signal.technical.rsi.btcusdt"
        assert TopicBuilder.sentiment_signal("twitter", "ETHUSDT") == "signal.sentiment.twitter.ethusdt"
    
    def test_trading_topics(self):
        """测试交易主题构建"""
        assert TopicBuilder.order_event("new", "BTCUSDT") == "order.new.btcusdt"
        assert TopicBuilder.position_update("ETHUSDT") == "position.update.ethusdt"
    
    def test_system_topics(self):
        """测试系统主题构建"""
        assert TopicBuilder.risk_alert("high", "BTCUSDT") == "risk.alert.high.btcusdt"
        assert TopicBuilder.risk_alert("critical") == "risk.alert.critical"
        assert TopicBuilder.system_event("start", "engine") == "system.start.engine"
        assert TopicBuilder.agent_event("register", "technical") == "agent.register.technical"


class TestMessageIntegration:
    """消息系统集成测试"""
    
    def test_message_to_topic_consistency(self):
        """测试消息到主题的一致性"""
        # 市场消息
        tick_msg = MarketTickMessage(
            symbol="BTCUSDT",
            price=50000.0,
            volume=1.0,
            bid_price=49999.0,
            ask_price=50001.0,
            bid_volume=5.0,
            ask_volume=5.0
        )
        assert tick_msg.to_topic() == TopicBuilder.market_tick("BTCUSDT")
        
        # 技术信号消息
        signal_msg = TechnicalSignalMessage(
            symbol="ETHUSDT",
            indicator="MACD",
            signal_type=SignalType.BUY,
            strength=0.8,
            confidence=0.9,
            price=3000.0,
            value=100.0
        )
        assert signal_msg.to_topic() == TopicBuilder.technical_signal("MACD", "ETHUSDT")
        
        # 订单消息
        order_msg = OrderMessage(
            order_id="123",
            client_order_id=None,
            symbol="ADAUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=100.0,
            price=None,
            status=OrderStatus.FILLED
        )
        assert order_msg.to_topic() == TopicBuilder.order_event("filled", "ADAUSDT")


if __name__ == "__main__":
    pytest.main([__file__])