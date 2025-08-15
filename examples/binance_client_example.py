#!/usr/bin/env python3
"""
币安期货客户端使用示例
演示基本的API调用功能
"""

import asyncio
import os
from src.exchanges.binance import BinanceFuturesClient, BinanceAPIError


async def main():
    """主函数 - 演示币安客户端基本用法"""
    
    # 从环境变量获取API凭证（可选）
    api_key = os.getenv('BINANCE_API_KEY', '')
    api_secret = os.getenv('BINANCE_API_SECRET', '')
    
    # 创建客户端（默认使用testnet）
    client = BinanceFuturesClient(
        api_key=api_key,
        api_secret=api_secret,
        testnet=True
    )
    
    try:
        async with client:
            print(f"Connected to Binance {client.environment}")
            
            # 1. 测试连接
            print("\n=== 测试连接 ===")
            ping_result = await client.ping()
            print(f"Ping: {ping_result}")
            
            # 2. 获取服务器时间
            server_time = await client.get_server_time()
            print(f"Server time: {server_time['serverTime']}")
            
            # 3. 获取交易规则
            print("\n=== 交易规则 ===")
            exchange_info = await client.get_exchange_info()
            symbols = [s['symbol'] for s in exchange_info['symbols'][:5]]
            print(f"Available symbols (first 5): {symbols}")
            
            # 4. 获取BTCUSDT价格
            print("\n=== 价格信息 ===")
            btc_price = await client.get_ticker_price('BTCUSDT')
            print(f"BTC/USDT price: ${btc_price['price']}")
            
            # 5. 获取24小时统计
            btc_24hr = await client.get_ticker_24hr('BTCUSDT')
            print(f"24h change: {btc_24hr['priceChangePercent']}%")
            
            # 6. 获取订单簿
            print("\n=== 订单簿 ===")
            order_book = await client.get_order_book('BTCUSDT', limit=5)
            print(f"Best bid: ${order_book['bids'][0][0]}")
            print(f"Best ask: ${order_book['asks'][0][0]}")
            
            # 7. 如果有API凭证，获取账户信息
            if api_key and api_secret:
                print("\n=== 账户信息 ===")
                try:
                    account_info = await client.get_account_info()
                    print(f"Total wallet balance: ${account_info['totalWalletBalance']}")
                    
                    # 获取余额
                    balances = await client.get_balance()
                    usdt_balance = next((b for b in balances if b['asset'] == 'USDT'), None)
                    if usdt_balance:
                        print(f"USDT balance: {usdt_balance['balance']}")
                        
                except BinanceAPIError as e:
                    print(f"API Error: {e}")
            else:
                print("\n=== 跳过私有API调用（需要API凭证）===")
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())