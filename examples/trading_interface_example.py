#!/usr/bin/env python3
"""
币安交易接口使用示例
演示模拟盘和实盘切换、数据隔离等功能
"""

import asyncio
import os
from datetime import datetime
from typing import Dict, Any

from src.exchanges import TradingManager, TradingEnvironment
from src.core.models import Order, OrderSide, OrderType, PositionSide, TimeInForce


async def demo_environment_switching():
    """演示环境切换功能"""
    manager = TradingManager()
    
    # 初始化交易接口
    await manager.initialize_binance_interfaces(
        api_key=os.getenv('BINANCE_API_KEY', ''),
        api_secret=os.getenv('BINANCE_API_SECRET', '')
    )
    
    print("🔄 环境切换演示")
    print(f"可用环境: {[env.value for env in manager.available_environments]}")
    
    # 演示testnet环境
    print("\n📊 切换到Testnet环境")
    async with manager.use_environment(TradingEnvironment.TESTNET) as testnet_interface:
        account_info = await testnet_interface.get_account_info()
        balance = await testnet_interface.get_balance()
        
        print(f"Testnet账户余额: {account_info.get('totalWalletBalance', 'N/A')} USDT")
        print(f"可用资金: {[b for b in balance if float(b.get('balance', 0)) > 0][:3]}")
    
    # 演示mainnet环境（需要真实API密钥）
    print("\n🏦 切换到Mainnet环境")
    try:
        async with manager.use_environment(TradingEnvironment.MAINNET) as mainnet_interface:
            # 只查询基本信息，不进行交易
            server_time = await mainnet_interface.client.get_server_time()
            print(f"Mainnet服务器时间: {datetime.fromtimestamp(server_time['serverTime']/1000)}")
    except Exception as e:
        print(f"Mainnet连接失败: {e}")


async def demo_data_isolation():
    """演示数据隔离功能"""
    manager = TradingManager()
    await manager.initialize_binance_interfaces()
    
    print("\n🔒 数据隔离演示")
    
    # 测试数据键隔离
    base_key = "user_orders"
    
    testnet_key = manager.get_isolated_key(base_key, TradingEnvironment.TESTNET)
    mainnet_key = manager.get_isolated_key(base_key, TradingEnvironment.MAINNET)
    
    print(f"基础键: {base_key}")
    print(f"Testnet隔离键: {testnet_key}")
    print(f"Mainnet隔离键: {mainnet_key}")
    
    # 测试数据过滤
    mock_data = [
        {"orderId": "1", "trading_environment": "testnet", "symbol": "BTCUSDT"},
        {"orderId": "2", "trading_environment": "mainnet", "symbol": "ETHUSDT"},
        {"orderId": "3", "trading_environment": "testnet", "symbol": "ADAUSDT"},
    ]
    
    testnet_filtered = manager.filter_data_by_environment(
        mock_data, TradingEnvironment.TESTNET
    )
    mainnet_filtered = manager.filter_data_by_environment(
        mock_data, TradingEnvironment.MAINNET
    )
    
    print(f"Testnet数据: {len(testnet_filtered)} 条")
    print(f"Mainnet数据: {len(mainnet_filtered)} 条")


async def demo_trading_operations():
    """演示交易操作"""
    manager = TradingManager()
    await manager.initialize_binance_interfaces()
    
    print("\n💱 交易操作演示 (Testnet)")
    
    # 设置为testnet环境
    manager.set_environment(TradingEnvironment.TESTNET)
    
    try:
        # 连接testnet
        connected = await manager.connect_environment(TradingEnvironment.TESTNET)
        if not connected:
            print("❌ 无法连接到Testnet")
            return
        
        # 查询账户信息
        account_info = await manager.get_account_info()
        print(f"账户总余额: {account_info.get('totalWalletBalance', 'N/A')} USDT")
        
        # 查询当前持仓
        positions = await manager.get_positions()
        active_positions = [p for p in positions if float(p.get('positionAmt', 0)) != 0]
        print(f"当前持仓: {len(active_positions)} 个")
        
        # 查询当前挂单
        open_orders = await manager.get_open_orders()
        print(f"当前挂单: {len(open_orders)} 个")
        
        # 演示下单（小额测试单）
        print("\n📈 演示下单流程")
        
        # 创建一个小额测试订单
        test_order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.001,  # 最小数量
            price=30000,     # 远低于市价，不会成交
            time_in_force=TimeInForce.GTC,
            position_side=PositionSide.BOTH
        )
        
        print(f"测试订单: {test_order.symbol} {test_order.side.value} {test_order.quantity} @ ${test_order.price}")
        
        # 注意：这里只是演示，实际运行时请谨慎
        # result = await manager.place_order(test_order)
        # print(f"订单结果: {result.get('orderId')}")
        print("(订单已跳过，避免实际交易)")
        
    except Exception as e:
        print(f"交易操作失败: {e}")
    finally:
        await manager.disconnect_environment(TradingEnvironment.TESTNET)


async def demo_cross_environment_comparison():
    """演示跨环境对比"""
    manager = TradingManager()
    await manager.initialize_binance_interfaces()
    
    print("\n🔍 跨环境对比演示")
    
    # 连接所有环境
    connection_results = await manager.connect_all()
    print(f"连接结果: {[(env.value, success) for env, success in connection_results.items()]}")
    
    try:
        # 比较账户信息
        print("\n💰 账户信息对比")
        account_comparison = await manager.compare_environments('get_account_info')
        
        for env, info in account_comparison.items():
            if 'error' not in info:
                balance = info.get('totalWalletBalance', 'N/A')
                print(f"{env.value}: {balance} USDT")
            else:
                print(f"{env.value}: 错误 - {info['error']}")
        
        # 比较持仓信息
        print("\n📊 持仓信息对比")
        positions_comparison = await manager.compare_environments('get_positions')
        
        for env, positions in positions_comparison.items():
            if 'error' not in positions:
                active_count = len([p for p in positions if float(p.get('positionAmt', 0)) != 0])
                print(f"{env.value}: {active_count} 个活跃持仓")
            else:
                print(f"{env.value}: 错误 - {positions['error']}")
    
    finally:
        await manager.disconnect_all()


async def main():
    """主函数"""
    print("🚀 币安交易接口演示程序")
    print("=" * 50)
    
    # 检查API凭证
    api_key = os.getenv('BINANCE_API_KEY', '')
    api_secret = os.getenv('BINANCE_API_SECRET', '')
    
    if not api_key or not api_secret:
        print("⚠️  警告: 未设置API凭证，部分功能将无法演示")
        print("请设置环境变量: BINANCE_API_KEY 和 BINANCE_API_SECRET")
    
    try:
        # 依次演示各个功能
        await demo_environment_switching()
        await demo_data_isolation()
        
        if api_key and api_secret:
            await demo_trading_operations()
            await demo_cross_environment_comparison()
        else:
            print("\n⏭️  跳过需要API凭证的演示")
    
    except KeyboardInterrupt:
        print("\n⏹️  程序被中断")
    except Exception as e:
        print(f"\n❌ 程序错误: {e}")
    
    print("\n✅ 演示完成")


if __name__ == "__main__":
    asyncio.run(main())