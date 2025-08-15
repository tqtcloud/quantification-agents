#!/usr/bin/env python3
"""
币安WebSocket市场数据订阅示例
演示如何订阅ticker、depth和kline数据流
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

from src.exchanges.binance import BinanceFuturesClient


async def ticker_handler(data: Dict[str, Any]):
    """处理ticker数据"""
    symbol = data.get('s')
    price = data.get('c')
    change_24h = data.get('P')
    timestamp = datetime.fromtimestamp(data.get('E', 0) / 1000)
    
    print(f"📈 TICKER | {symbol} | ${price} | 24h: {change_24h}% | {timestamp}")


async def depth_handler(data: Dict[str, Any]):
    """处理订单簿数据"""
    symbol = data.get('s')
    bids = data.get('b', [])
    asks = data.get('a', [])
    
    if bids and asks:
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        spread = best_ask - best_bid
        spread_pct = (spread / best_bid) * 100
        
        print(f"📊 DEPTH | {symbol} | Bid: ${best_bid} | Ask: ${best_ask} | Spread: {spread_pct:.4f}%")


async def kline_handler(data: Dict[str, Any]):
    """处理K线数据"""
    kline_data = data.get('k', {})
    if not kline_data:
        return
        
    symbol = kline_data.get('s')
    open_price = kline_data.get('o')
    high_price = kline_data.get('h')
    low_price = kline_data.get('l')
    close_price = kline_data.get('c')
    volume = kline_data.get('v')
    is_closed = kline_data.get('x', False)  # K线是否已完成
    
    status = "CLOSED" if is_closed else "LIVE"
    print(f"📊 KLINE | {symbol} | O: ${open_price} | H: ${high_price} | L: ${low_price} | C: ${close_price} | Vol: {volume} | {status}")


async def main():
    """主函数 - 演示WebSocket数据订阅"""
    
    # 创建客户端（使用testnet）
    client = BinanceFuturesClient(testnet=True)
    
    try:
        async with client:
            print("🚀 Connected to Binance WebSocket streams")
            print("Press Ctrl+C to stop...\n")
            
            # 订阅多个数据流
            tasks = []
            
            # 1. 订阅BTCUSDT ticker
            print("📈 Subscribing to BTCUSDT ticker...")
            ticker_task = await client.subscribe_ticker('BTCUSDT', ticker_handler)
            tasks.append(ticker_task)
            
            # 2. 订阅BTCUSDT 深度数据
            print("📊 Subscribing to BTCUSDT depth...")
            depth_task = await client.subscribe_depth('BTCUSDT', depth_handler, speed='100ms')
            tasks.append(depth_task)
            
            # 3. 订阅BTCUSDT 1分钟K线
            print("📈 Subscribing to BTCUSDT 1m klines...")
            kline_task = await client.subscribe_kline('BTCUSDT', '1m', kline_handler)
            tasks.append(kline_task)
            
            # 显示活跃的数据流
            await asyncio.sleep(1)  # 等待连接建立
            active_streams = client.active_streams
            print(f"\n✅ Active streams: {len(active_streams)}")
            for stream in active_streams:
                print(f"   - {stream}")
            print()
            
            # 等待所有任务完成（或被中断）
            try:
                await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                print("\n⏹️  Tasks cancelled")
    
    except KeyboardInterrupt:
        print("\n⏹️  Received interrupt signal")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        print("🔌 Closing WebSocket connections...")


async def single_stream_example():
    """单个数据流示例"""
    client = BinanceFuturesClient(testnet=True)
    
    async def simple_ticker_handler(data: Dict[str, Any]):
        symbol = data.get('s')
        price = data.get('c')
        print(f"{symbol}: ${price}")
    
    try:
        # 订阅单个ticker
        task = await client.subscribe_ticker('ETHUSDT', simple_ticker_handler)
        
        # 运行30秒
        await asyncio.sleep(30)
        
        # 手动取消订阅
        task.cancel()
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.close_all_websockets()


if __name__ == "__main__":
    print("选择示例:")
    print("1. 多数据流示例 (ticker + depth + kline)")
    print("2. 单数据流示例 (ticker only)")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "2":
        asyncio.run(single_stream_example())
    else:
        asyncio.run(main())