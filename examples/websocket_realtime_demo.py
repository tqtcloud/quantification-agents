"""
WebSocket实时推送系统演示

展示如何使用WebSocket系统进行实时数据推送，
包括交易信号、策略状态、风险警报等。
"""

import asyncio
import json
import logging
import websockets
from datetime import datetime
from typing import Dict, Any
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.websocket import (
    WebSocketManager,
    WebSocketConfig,
    MessageType,
    SubscriptionType
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WebSocketDemo:
    """WebSocket演示"""
    
    def __init__(self):
        # WebSocket配置
        self.config = WebSocketConfig(
            host="localhost",
            port=8765,
            max_connections=100,
            ping_interval=30,
            connection_timeout=300,
            auth_required=False,  # 演示模式不需要认证
            compression_enabled=True
        )
        
        # WebSocket管理器
        self.ws_manager = WebSocketManager(self.config)
        
        # 演示数据
        self.demo_running = False
        self.demo_task = None
    
    async def start_server(self):
        """启动WebSocket服务器"""
        logger.info("启动WebSocket服务器...")
        
        # 启动WebSocket管理器
        await self.ws_manager.start()
        
        logger.info(f"WebSocket服务器启动成功，监听 {self.config.host}:{self.config.port}")
        
        # 启动演示数据推送
        self.demo_running = True
        self.demo_task = asyncio.create_task(self._demo_data_loop())
    
    async def stop_server(self):
        """停止WebSocket服务器"""
        logger.info("停止WebSocket服务器...")
        
        # 停止演示数据推送
        self.demo_running = False
        if self.demo_task:
            self.demo_task.cancel()
            try:
                await self.demo_task
            except asyncio.CancelledError:
                pass
        
        # 停止WebSocket管理器
        await self.ws_manager.stop()
        
        logger.info("WebSocket服务器已停止")
    
    async def _demo_data_loop(self):
        """演示数据推送循环"""
        logger.info("启动演示数据推送...")
        
        counter = 0
        
        while self.demo_running:
            try:
                counter += 1
                
                # 每5秒推送一次交易信号
                if counter % 5 == 0:
                    await self._push_trading_signal()
                
                # 每10秒推送一次策略状态
                if counter % 10 == 0:
                    await self._push_strategy_status()
                
                # 每15秒推送一次市场数据
                if counter % 15 == 0:
                    await self._push_market_data()
                
                # 每30秒推送一次系统监控
                if counter % 30 == 0:
                    await self._push_system_monitor()
                
                # 模拟风险警报（随机触发）
                if counter % 60 == 0:
                    await self._push_risk_alert()
                
                # 每秒执行一次
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"演示数据推送异常: {e}")
                await asyncio.sleep(1)
        
        logger.info("演示数据推送已停止")
    
    async def _push_trading_signal(self):
        """推送交易信号"""
        import random
        
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT"]
        signals = ["BUY", "SELL", "HOLD"]
        
        signal_data = {
            "signal_id": f"sig_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "symbol": random.choice(symbols),
            "signal": random.choice(signals),
            "price": round(random.uniform(30000, 70000), 2),
            "confidence": round(random.uniform(0.6, 0.95), 2),
            "strategy": "momentum_breakout",
            "timeframe": "1h",
            "timestamp": datetime.now().isoformat()
        }
        
        success_count = await self.ws_manager.broadcast_trading_signal(signal_data)
        logger.info(f"推送交易信号到 {success_count} 个连接: {signal_data['symbol']} {signal_data['signal']}")
    
    async def _push_strategy_status(self):
        """推送策略状态"""
        import random
        
        strategies = ["momentum_strategy", "mean_reversion", "arbitrage_bot"]
        
        for strategy_id in strategies:
            status_data = {
                "strategy_id": strategy_id,
                "status": random.choice(["running", "paused", "stopped"]),
                "performance": {
                    "total_pnl": round(random.uniform(-1000, 5000), 2),
                    "daily_pnl": round(random.uniform(-500, 1000), 2),
                    "win_rate": round(random.uniform(0.4, 0.8), 2),
                    "sharpe_ratio": round(random.uniform(0.5, 2.5), 2),
                    "max_drawdown": round(random.uniform(0.05, 0.25), 2)
                },
                "positions": {
                    "active_positions": random.randint(0, 5),
                    "total_exposure": round(random.uniform(10000, 100000), 2)
                },
                "last_signal_time": datetime.now().isoformat(),
                "uptime_hours": random.randint(1, 168)
            }
            
            success_count = await self.ws_manager.broadcast_strategy_status(strategy_id, status_data)
            logger.info(f"推送策略状态到 {success_count} 个连接: {strategy_id}")
    
    async def _push_market_data(self):
        """推送市场数据"""
        import random
        
        market_data = {
            "data_type": "kline",
            "symbol": "BTCUSDT",
            "timeframe": "1m",
            "data": {
                "open": round(random.uniform(45000, 55000), 2),
                "high": round(random.uniform(55000, 60000), 2),
                "low": round(random.uniform(40000, 45000), 2),
                "close": round(random.uniform(45000, 55000), 2),
                "volume": round(random.uniform(100, 1000), 2),
                "timestamp": datetime.now().isoformat()
            },
            "indicators": {
                "rsi": round(random.uniform(20, 80), 2),
                "macd": round(random.uniform(-500, 500), 2),
                "bb_upper": round(random.uniform(55000, 60000), 2),
                "bb_lower": round(random.uniform(40000, 45000), 2)
            }
        }
        
        success_count = await self.ws_manager.broadcast_market_data(market_data)
        logger.info(f"推送市场数据到 {success_count} 个连接")
    
    async def _push_system_monitor(self):
        """推送系统监控数据"""
        import random, psutil
        
        # 获取实际系统指标（如果可用）
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
        except:
            # 模拟数据
            cpu_percent = random.uniform(10, 80)
            memory = type('obj', (object,), {'percent': random.uniform(40, 90)})
            disk = type('obj', (object,), {'percent': random.uniform(20, 70)})
        
        monitor_data = {
            "system": {
                "cpu_percent": round(cpu_percent, 1),
                "memory_percent": round(memory.percent, 1),
                "disk_percent": round(disk.percent, 1),
                "uptime_seconds": random.randint(3600, 86400)
            },
            "websocket": self.ws_manager.get_system_stats(),
            "trading": {
                "active_orders": random.randint(0, 20),
                "pending_signals": random.randint(0, 5),
                "api_calls_per_minute": random.randint(50, 200),
                "latency_ms": round(random.uniform(10, 100), 1)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        success_count = await self.ws_manager.broadcast_system_monitor(monitor_data)
        logger.info(f"推送系统监控数据到 {success_count} 个连接")
    
    async def _push_risk_alert(self):
        """推送风险警报"""
        import random
        
        alert_types = [
            "position_limit_exceeded",
            "high_volatility_detected",
            "correlation_risk",
            "liquidity_warning",
            "drawdown_limit"
        ]
        
        severities = ["low", "medium", "high", "critical"]
        
        alert_data = {
            "alert_id": f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "alert_type": random.choice(alert_types),
            "severity": random.choice(severities),
            "symbol": random.choice(["BTCUSDT", "ETHUSDT", "Portfolio"]),
            "message": "检测到异常风险条件，请及时处理",
            "details": {
                "current_value": round(random.uniform(1000, 10000), 2),
                "threshold": round(random.uniform(500, 5000), 2),
                "breach_percentage": round(random.uniform(10, 50), 1)
            },
            "recommended_action": "减少仓位或暂停交易",
            "timestamp": datetime.now().isoformat()
        }
        
        success_count = await self.ws_manager.broadcast_risk_alert(alert_data)
        logger.info(f"推送风险警报到 {success_count} 个连接: {alert_data['alert_type']} ({alert_data['severity']})")
    
    async def run_forever(self):
        """持续运行服务器"""
        try:
            await self.start_server()
            
            logger.info("WebSocket服务器运行中，按Ctrl+C停止...")
            
            # 等待中断信号
            await self.ws_manager.wait_until_stopped()
            
        except KeyboardInterrupt:
            logger.info("收到中断信号，正在停止服务器...")
        except Exception as e:
            logger.error(f"服务器运行异常: {e}")
        finally:
            await self.stop_server()


class WebSocketClient:
    """WebSocket客户端演示"""
    
    def __init__(self, server_url="ws://localhost:8765"):
        self.server_url = server_url
        self.websocket = None
        self.subscriptions = set()
    
    async def connect(self):
        """连接到WebSocket服务器"""
        logger.info(f"连接到WebSocket服务器: {self.server_url}")
        
        try:
            self.websocket = await websockets.connect(self.server_url)
            logger.info("连接成功")
            
            # 启动消息接收任务
            receive_task = asyncio.create_task(self._receive_messages())
            
            return receive_task
            
        except Exception as e:
            logger.error(f"连接失败: {e}")
            raise
    
    async def disconnect(self):
        """断开连接"""
        if self.websocket:
            await self.websocket.close()
            logger.info("连接已断开")
    
    async def subscribe(self, subscription_type: str, filters: Dict[str, Any] = None):
        """订阅数据类型"""
        if not self.websocket:
            logger.error("未连接到服务器")
            return
        
        message = {
            "type": "subscribe",
            "data": {
                "type": subscription_type,
                "filters": filters or {}
            }
        }
        
        await self.websocket.send(json.dumps(message))
        self.subscriptions.add(subscription_type)
        logger.info(f"订阅: {subscription_type}")
    
    async def unsubscribe(self, subscription_type: str):
        """取消订阅"""
        if not self.websocket:
            logger.error("未连接到服务器")
            return
        
        message = {
            "type": "unsubscribe",
            "data": {
                "type": subscription_type
            }
        }
        
        await self.websocket.send(json.dumps(message))
        self.subscriptions.discard(subscription_type)
        logger.info(f"取消订阅: {subscription_type}")
    
    async def _receive_messages(self):
        """接收消息"""
        logger.info("开始接收消息...")
        
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError:
                    logger.error(f"无法解析消息: {message}")
                except Exception as e:
                    logger.error(f"处理消息异常: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("连接已关闭")
        except Exception as e:
            logger.error(f"接收消息异常: {e}")
    
    async def _handle_message(self, data: Dict[str, Any]):
        """处理收到的消息"""
        message_type = data.get("type", "unknown")
        
        if message_type == "ping":
            logger.debug("收到PING，发送PONG")
            await self.websocket.send(json.dumps({"type": "pong"}))
        
        elif message_type == "trading_signal":
            signal = data.get("data", {}).get("signal", {})
            logger.info(f"收到交易信号: {signal.get('symbol')} {signal.get('signal')} @ {signal.get('price')}")
        
        elif message_type == "strategy_status":
            status = data.get("data", {}).get("status", {})
            strategy_id = data.get("data", {}).get("strategy_id")
            pnl = status.get("performance", {}).get("total_pnl", 0)
            logger.info(f"收到策略状态: {strategy_id} PNL={pnl}")
        
        elif message_type == "market_data":
            market_data = data.get("data", {}).get("data", {})
            symbol = data.get("data", {}).get("symbol")
            price = market_data.get("close", 0)
            logger.info(f"收到市场数据: {symbol} 价格={price}")
        
        elif message_type == "risk_alert":
            alert = data.get("data", {}).get("alert", {})
            alert_type = alert.get("alert_type")
            severity = alert.get("severity")
            logger.warning(f"收到风险警报: {alert_type} ({severity})")
        
        elif message_type == "system_monitor":
            metrics = data.get("data", {}).get("metrics", {})
            cpu = metrics.get("system", {}).get("cpu_percent", 0)
            logger.info(f"收到系统监控: CPU={cpu}%")
        
        elif message_type in ["subscribe", "unsubscribe"]:
            success = data.get("data", {}).get("success", False)
            logger.info(f"{message_type} 响应: {'成功' if success else '失败'}")
        
        elif message_type == "error":
            error = data.get("data", {}).get("error", "未知错误")
            logger.error(f"收到错误消息: {error}")
        
        else:
            logger.debug(f"收到未知消息类型: {message_type}")


async def run_server_demo():
    """运行服务器演示"""
    demo = WebSocketDemo()
    await demo.run_forever()


async def run_client_demo():
    """运行客户端演示"""
    client = WebSocketClient()
    
    try:
        # 连接服务器
        receive_task = await client.connect()
        
        # 订阅各种数据类型
        await client.subscribe("trading_signals", {"symbol": "BTCUSDT"})
        await client.subscribe("strategy_status")
        await client.subscribe("market_data", {"symbol": "BTCUSDT"})
        await client.subscribe("risk_alerts")
        await client.subscribe("system_monitor")
        
        # 运行60秒
        await asyncio.sleep(60)
        
        # 取消一些订阅
        await client.unsubscribe("market_data")
        await client.unsubscribe("system_monitor")
        
        # 再运行30秒
        await asyncio.sleep(30)
        
    except KeyboardInterrupt:
        logger.info("客户端演示中断")
    finally:
        await client.disconnect()


async def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
    else:
        print("使用方法:")
        print("  python websocket_realtime_demo.py server   # 运行服务器")
        print("  python websocket_realtime_demo.py client   # 运行客户端")
        return
    
    if mode == "server":
        await run_server_demo()
    elif mode == "client":
        await run_client_demo()
    else:
        print(f"未知模式: {mode}")


if __name__ == "__main__":
    asyncio.run(main())