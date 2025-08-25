#!/usr/bin/env python3
"""
多Agent智能分析系统综合演示

展示完整的17-Agent协作投资决策流程：
1. 数据预处理
2. 15个分析师Agent并行分析
3. 风险管理Agent评估
4. 投资组合管理Agent优化
5. 最终投资决策输出

使用方法：
python examples/multi_agent_system_demo.py
"""

import asyncio
import sys
import os
from typing import Dict, Any
import json
from datetime import datetime, timedelta

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.orchestrator import MultiAgentOrchestrator, WorkflowConfig
from src.agents.models import AgentState
from src.utils.logger import setup_logging

# 设置日志
logger = setup_logging("multi_agent_demo", log_level="INFO")


class MultiAgentSystemDemo:
    """多Agent智能分析系统演示"""
    
    def __init__(self):
        """初始化演示系统"""
        self.orchestrator = None
        self.demo_data = self._generate_demo_data()
        
        logger.info("🚀 多Agent智能分析系统演示初始化完成")
    
    def _generate_demo_data(self) -> Dict[str, Any]:
        """生成演示用的市场数据"""
        return {
            'symbols': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
            'prices': {
                'BTCUSDT': {
                    'current': 45000.0,
                    'high_24h': 46500.0,
                    'low_24h': 44200.0,
                    'volume_24h': 1500000000,
                    'price_change_24h': 2.5
                },
                'ETHUSDT': {
                    'current': 2800.0,
                    'high_24h': 2850.0,
                    'low_24h': 2750.0,
                    'volume_24h': 800000000,
                    'price_change_24h': 1.8
                },
                'BNBUSDT': {
                    'current': 320.0,
                    'high_24h': 325.0,
                    'low_24h': 315.0,
                    'volume_24h': 150000000,
                    'price_change_24h': -0.5
                }
            },
            'technical_indicators': {
                'BTCUSDT': {
                    'rsi': 62.5,
                    'macd': 150.2,
                    'bollinger_upper': 46200.0,
                    'bollinger_lower': 43800.0,
                    'ema_20': 44800.0,
                    'ema_50': 44200.0
                },
                'ETHUSDT': {
                    'rsi': 58.3,
                    'macd': 25.8,
                    'bollinger_upper': 2820.0,
                    'bollinger_lower': 2780.0,
                    'ema_20': 2790.0,
                    'ema_50': 2760.0
                }
            },
            'market_sentiment': {
                'fear_greed_index': 72,  # 贪婪
                'social_sentiment': 0.6,
                'news_sentiment': 0.4,
                'options_put_call_ratio': 0.8
            },
            'macro_data': {
                'inflation_rate': 3.2,
                'unemployment_rate': 3.8,
                'fed_funds_rate': 5.25,
                'gdp_growth': 2.1,
                'dxy_index': 103.5
            },
            'news': [
                "美联储官员暗示可能暂停加息",
                "比特币ETF申请获得新进展",
                "以太坊升级成功完成",
                "加密货币监管框架即将出台"
            ]
        }
    
    async def setup_orchestrator(self):
        """设置工作流编排器"""
        config = WorkflowConfig(
            max_parallel_agents=6,
            node_timeout=60.0,
            workflow_timeout=300.0,
            enable_retry=True,
            max_retries=2,
            aggregation_method='weighted_voting',
            risk_threshold=0.7,
            min_confidence=0.6
        )
        
        self.orchestrator = MultiAgentOrchestrator(config)
        await self.orchestrator.initialize()
        
        logger.info("✅ 工作流编排器初始化完成")
    
    def _prepare_agent_state(self) -> AgentState:
        """准备Agent状态数据"""
        return AgentState(
            market_data=self.demo_data,
            news_data=self.demo_data['news'],
            social_sentiment=self.demo_data['market_sentiment'],
            analyst_opinions=[],
            confidence_scores={},
            risk_assessment=None,
            portfolio_recommendations=None,
            final_decision=None,
            reasoning_chain=[],
            created_at=datetime.now(),
            updated_at=datetime.now(),
            agent_id="demo_orchestrator",
            session_id="demo_session_001",
            version=1
        )
    
    async def run_analysis_workflow(self) -> Dict[str, Any]:
        """运行完整的分析工作流"""
        logger.info("🎯 开始运行17-Agent智能分析工作流")
        
        # 准备初始状态
        initial_state = self._prepare_agent_state()
        
        # 设置进度回调
        def progress_callback(node_name: str, status: str, data: Any = None):
            if status == "started":
                logger.info(f"🔄 开始执行: {node_name}")
            elif status == "completed":
                logger.info(f"✅ 完成执行: {node_name}")
            elif status == "failed":
                logger.error(f"❌ 执行失败: {node_name}")
        
        self.orchestrator.add_callback('node_progress', progress_callback)
        
        try:
            # 执行工作流
            start_time = datetime.now()
            result = await self.orchestrator.execute_workflow(initial_state)
            execution_time = datetime.now() - start_time
            
            logger.info(f"⏱️ 工作流执行耗时: {execution_time.total_seconds():.2f}秒")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 工作流执行失败: {e}")
            return None
    
    def _analyze_results(self, results: Dict[str, Any]) -> None:
        """分析和展示结果"""
        if not results:
            logger.error("❌ 没有分析结果可展示")
            return
        
        logger.info("📊 智能投资分析结果")
        logger.info("=" * 60)
        
        # 分析师观点统计
        analyst_opinions = results.get('analyst_opinions', [])
        if analyst_opinions:
            logger.info(f"👥 分析师观点: {len(analyst_opinions)}个")
            
            # 统计投资建议分布
            buy_count = sum(1 for op in analyst_opinions if op.get('recommendation') == 'BUY')
            hold_count = sum(1 for op in analyst_opinions if op.get('recommendation') == 'HOLD')
            sell_count = sum(1 for op in analyst_opinions if op.get('recommendation') == 'SELL')
            
            logger.info(f"   📈 买入建议: {buy_count}")
            logger.info(f"   📊 持有建议: {hold_count}")  
            logger.info(f"   📉 卖出建议: {sell_count}")
            
            # 平均置信度
            avg_confidence = sum(op.get('confidence', 0) for op in analyst_opinions) / len(analyst_opinions)
            logger.info(f"   🎯 平均置信度: {avg_confidence:.2f}")
        
        # 风险评估
        risk_assessment = results.get('risk_assessment')
        if risk_assessment:
            logger.info("⚠️ 风险评估:")
            logger.info(f"   📊 整体风险评分: {risk_assessment.get('overall_risk_score', 0):.2f}")
            logger.info(f"   💰 建议仓位大小: {risk_assessment.get('recommended_position_size', 0):.2f}")
            logger.info(f"   🛑 止损水平: {risk_assessment.get('stop_loss_level', 0):.2f}")
        
        # 投资组合建议
        portfolio_recommendations = results.get('portfolio_recommendations')
        if portfolio_recommendations:
            logger.info("💼 投资组合建议:")
            logger.info(f"   📊 推荐操作: {portfolio_recommendations.get('action', 'UNKNOWN')}")
            logger.info(f"   🎯 置信度: {portfolio_recommendations.get('confidence', 0):.2f}")
            
            allocation = portfolio_recommendations.get('allocation', {})
            if allocation:
                logger.info("   💰 资产配置:")
                for asset, weight in allocation.items():
                    logger.info(f"      {asset}: {weight:.2%}")
        
        # 最终决策
        final_decision = results.get('final_decision')
        if final_decision:
            logger.info(f"🎯 最终投资决策: {final_decision}")
        
        # 推理链
        reasoning_chain = results.get('reasoning_chain', [])
        if reasoning_chain:
            logger.info("🔍 关键推理步骤:")
            for i, reason in enumerate(reasoning_chain[-3:], 1):  # 显示最后3步
                logger.info(f"   {i}. {reason}")
    
    def _display_agent_statistics(self) -> None:
        """显示Agent执行统计"""
        if not self.orchestrator:
            return
        
        stats = self.orchestrator.get_execution_stats()
        
        logger.info("📈 Agent执行统计")
        logger.info("=" * 40)
        logger.info(f"总执行次数: {stats.get('total_executions', 0)}")
        logger.info(f"成功次数: {stats.get('successful_executions', 0)}")
        logger.info(f"失败次数: {stats.get('failed_executions', 0)}")
        logger.info(f"平均执行时间: {stats.get('average_execution_time', 0):.2f}秒")
        
        # 节点统计
        node_stats = stats.get('node_stats', {})
        if node_stats:
            logger.info("各节点执行情况:")
            for node_name, node_stat in node_stats.items():
                logger.info(f"  {node_name}: {node_stat.get('executions', 0)}次")
    
    async def run_demo(self):
        """运行完整演示"""
        logger.info("🎬 开始多Agent智能分析系统综合演示")
        logger.info("=" * 80)
        
        try:
            # 1. 设置编排器
            await self.setup_orchestrator()
            
            # 2. 运行分析工作流
            results = await self.run_analysis_workflow()
            
            # 3. 分析结果
            if results:
                self._analyze_results(results)
            
            # 4. 显示统计信息
            self._display_agent_statistics()
            
            logger.info("=" * 80)
            logger.info("🎉 多Agent智能分析系统演示完成！")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 演示过程中发生错误: {e}")
            return None


async def main():
    """主函数"""
    print("🎯 量化交易系统 - 多Agent智能分析演示")
    print("=" * 60)
    print("该演示将展示17个Agent的协作投资决策过程：")
    print("1. 数据预处理和环境准备")
    print("2. 15个投资大师分析师并行分析")
    print("3. 风险管理Agent评估风险")
    print("4. 投资组合Agent优化配置")
    print("5. 生成最终投资决策")
    print()
    
    # 创建并运行演示
    demo = MultiAgentSystemDemo()
    results = await demo.run_demo()
    
    if results:
        print("✅ 演示成功完成！")
        print("💡 系统已生成智能投资建议")
    else:
        print("❌ 演示过程中出现问题")
    
    return results


if __name__ == "__main__":
    # 设置事件循环策略 (Windows兼容)
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # 运行演示
    asyncio.run(main())