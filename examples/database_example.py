#!/usr/bin/env python3
"""
数据库管理器使用示例
演示多环境数据隔离、连接池管理和数据访问层
"""

import asyncio
from datetime import datetime
from pathlib import Path

from src.core.database import (
    db_manager,
    session_repo,
    order_repo,
    TradingSession,
    OrderRecord
)
from src.exchanges.trading_interface import TradingEnvironment
from src.core.models import OrderSide, OrderType, OrderStatus


def demo_environment_isolation():
    """演示环境数据隔离"""
    print("🔒 环境数据隔离演示")
    print("=" * 50)
    
    # 初始化所有环境的数据库
    db_manager.init_all_databases()
    
    # 创建不同环境的交易会话
    environments = [TradingEnvironment.TESTNET, TradingEnvironment.MAINNET, TradingEnvironment.PAPER]
    
    for env in environments:
        db_manager.set_current_environment(env)
        
        # 创建交易会话
        session_data = {
            "session_id": f"demo_session_{env.value}",
            "mode": "demo",
            "initial_balance": 10000.0
        }
        
        trading_session = session_repo.create_session(session_data, env)
        print(f"✅ Created session for {env.value}: {trading_session.session_id}")
        
        # 创建订单记录
        order_data = {
            "session_id": trading_session.id,
            "order_id": f"order_{env.value}_001",
            "symbol": "BTCUSDT",
            "side": OrderSide.BUY,
            "order_type": OrderType.LIMIT,
            "quantity": 0.1,
            "price": 50000.0,
            "status": OrderStatus.NEW
        }
        
        order = order_repo.create_order(order_data, env)
        print(f"✅ Created order for {env.value}: {order.order_id}")
    
    print()


def demo_data_access_isolation():
    """演示数据访问隔离"""
    print("📊 数据访问隔离演示")
    print("=" * 50)
    
    for env in TradingEnvironment:
        print(f"\n{env.value.upper()} 环境数据:")
        
        # 获取该环境的所有会话
        sessions = session_repo.get_sessions_by_environment(env)
        print(f"  - 交易会话数: {len(sessions)}")
        
        # 获取该环境的所有订单
        orders = order_repo.get_orders_by_environment(env)
        print(f"  - 订单数: {len(orders)}")
        
        if orders:
            print(f"  - 订单示例: {orders[0].order_id}")
    
    print()


def demo_connection_pooling():
    """演示连接池管理"""
    print("🔗 连接池管理演示")
    print("=" * 50)
    
    # 检查各环境的连接状态
    for env in TradingEnvironment:
        engine = db_manager.get_engine(env)
        print(f"✅ {env.value} 环境引擎: {engine}")
        
        # 测试连接
        with db_manager.get_session(env) as session:
            result = session.execute("SELECT 1").scalar()
            print(f"   连接测试: {result == 1}")
    
    print()


def demo_environment_switching():
    """演示环境切换"""
    print("🔄 环境切换演示")
    print("=" * 50)
    
    # 当前环境
    current = db_manager.get_current_environment()
    print(f"当前环境: {current.value if current else 'None'}")
    
    # 切换到不同环境
    for env in TradingEnvironment:
        db_manager.set_current_environment(env)
        print(f"切换到: {env.value}")
        
        # 在当前环境中获取活跃会话
        active_session = session_repo.get_active_session()
        print(f"  活跃会话: {active_session.session_id if active_session else 'None'}")
    
    # 使用临时环境切换
    print("\n使用临时环境切换:")
    db_manager.set_current_environment(TradingEnvironment.TESTNET)
    print(f"外部环境: {db_manager.get_current_environment().value}")
    
    with db_manager.use_environment(TradingEnvironment.MAINNET):
        print(f"临时环境: {db_manager.get_current_environment().value}")
    
    print(f"恢复环境: {db_manager.get_current_environment().value}")
    print()


def demo_database_structure():
    """演示数据库结构"""
    print("🏗️ 数据库结构演示")
    print("=" * 50)
    
    data_dir = Path("./data")
    print(f"数据根目录: {data_dir.absolute()}")
    
    for env in TradingEnvironment:
        env_dir = data_dir / env.value
        db_file = env_dir / f"trading_{env.value}.db"
        
        print(f"\n{env.value.upper()} 环境:")
        print(f"  目录: {env_dir}")
        print(f"  数据库: {db_file}")
        print(f"  存在: {db_file.exists()}")
        if db_file.exists():
            size = db_file.stat().st_size
            print(f"  大小: {size} bytes")
    
    print()


def demo_transaction_management():
    """演示事务管理"""
    print("🔄 事务管理演示")
    print("=" * 50)
    
    env = TradingEnvironment.TESTNET
    db_manager.set_current_environment(env)
    
    try:
        # 正常事务
        print("✅ 正常事务提交:")
        with db_manager.get_session(env) as session:
            session_data = TradingSession(
                session_id="transaction_test_1",
                environment=env,
                mode="test",
                initial_balance=5000.0
            )
            session.add(session_data)
            # 自动提交
        
        print("   事务已提交")
        
        # 异常事务回滚
        print("\n❌ 异常事务回滚:")
        try:
            with db_manager.get_session(env) as session:
                session_data = TradingSession(
                    session_id="transaction_test_2",
                    environment=env,
                    mode="test",
                    initial_balance=5000.0
                )
                session.add(session_data)
                # 故意引发异常
                raise ValueError("测试异常")
        except ValueError as e:
            print(f"   捕获异常: {e}")
            print("   事务已回滚")
        
        # 验证回滚效果
        with db_manager.get_session(env) as session:
            count = session.query(TradingSession).filter_by(
                session_id="transaction_test_2"
            ).count()
            print(f"   验证回滚: 记录数 = {count} (应为0)")
    
    except Exception as e:
        print(f"事务管理演示出错: {e}")
    
    print()


def main():
    """主函数"""
    print("🚀 数据库管理器演示程序")
    print("=" * 50)
    
    try:
        # 依次演示各项功能
        demo_environment_isolation()
        demo_data_access_isolation()
        demo_connection_pooling()
        demo_environment_switching()
        demo_database_structure()
        demo_transaction_management()
        
        print("✅ 所有演示完成")
        
    except Exception as e:
        print(f"❌ 演示出错: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        db_manager.close_all()
        print("🔒 已关闭所有数据库连接")


if __name__ == "__main__":
    main()