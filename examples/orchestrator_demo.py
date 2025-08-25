"""
MultiAgentOrchestrator演示
展示如何使用LangGraph工作流编排系统
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any

from src.agents.orchestrator import MultiAgentOrchestrator, WorkflowConfig
from src.agents.models import AgentState, MarketDataState, NewsDataState
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_market_data() -> Dict[str, MarketDataState]:
    """创建示例市场数据"""
    return {
        "BTC/USDT": MarketDataState(
            symbol="BTC/USDT",
            price=52000.0,
            volume=15000.0,
            bid=51950.0,
            ask=52050.0,
            spread=100.0,
            volatility=0.028,
            trend="up",
            momentum=0.82,
            timestamp=datetime.now(),
            metadata={
                "exchange": "Binance",
                "24h_change": 0.05,
                "24h_high": 53000.0,
                "24h_low": 49500.0
            }
        ),
        "ETH/USDT": MarketDataState(
            symbol="ETH/USDT",
            price=3200.0,
            volume=8000.0,
            bid=3195.0,
            ask=3205.0,
            spread=10.0,
            volatility=0.032,
            trend="up",
            momentum=0.75,
            timestamp=datetime.now(),
            metadata={
                "exchange": "Binance",
                "24h_change": 0.03,
                "24h_high": 3300.0,
                "24h_low": 3100.0
            }
        ),
        "SOL/USDT": MarketDataState(
            symbol="SOL/USDT",
            price=120.0,
            volume=5000.0,
            bid=119.5,
            ask=120.5,
            spread=1.0,
            volatility=0.045,
            trend="sideways",
            momentum=0.5,
            timestamp=datetime.now(),
            metadata={
                "exchange": "Binance",
                "24h_change": -0.01,
                "24h_high": 125.0,
                "24h_low": 118.0
            }
        )
    }


def create_sample_news_data() -> list:
    """创建示例新闻数据"""
    return [
        NewsDataState(
            source="Bloomberg",
            title="Bitcoin ETF见证创纪录资金流入",
            content="机构投资者持续增加比特币ETF配置，昨日净流入超过5亿美元...",
            sentiment_score=0.85,
            relevance_score=0.95,
            impact_level="high",
            entities=["Bitcoin", "ETF", "BlackRock", "Grayscale"],
            timestamp=datetime.now(),
            url="https://example.com/btc-etf-inflow"
        ),
        NewsDataState(
            source="CoinDesk",
            title="以太坊Layer 2生态系统持续扩张",
            content="Arbitrum和Optimism的总锁仓价值创新高，Layer 2采用率加速...",
            sentiment_score=0.7,
            relevance_score=0.8,
            impact_level="medium",
            entities=["Ethereum", "Layer 2", "Arbitrum", "Optimism"],
            timestamp=datetime.now(),
            url="https://example.com/eth-l2-growth"
        ),
        NewsDataState(
            source="Reuters",
            title="美联储暗示可能放缓加息步伐",
            content="美联储官员表示通胀数据显示积极信号，可能考虑放缓加息...",
            sentiment_score=0.6,
            relevance_score=0.7,
            impact_level="high",
            entities=["Federal Reserve", "Interest Rates", "Inflation"],
            timestamp=datetime.now(),
            url="https://example.com/fed-rates"
        )
    ]


def create_sample_social_sentiment() -> Dict[str, Any]:
    """创建示例社交情绪数据"""
    return {
        "BTC/USDT": {
            "platform": "twitter",
            "symbol": "BTC/USDT",
            "sentiment_score": 0.75,
            "volume": 25000,
            "trending_score": 0.9,
            "key_topics": ["ETF", "Halving", "Institutional"],
            "influencer_sentiment": 0.82,
            "retail_sentiment": 0.68,
            "timestamp": datetime.now()
        },
        "ETH/USDT": {
            "platform": "reddit",
            "symbol": "ETH/USDT",
            "sentiment_score": 0.65,
            "volume": 15000,
            "trending_score": 0.7,
            "key_topics": ["DeFi", "Layer2", "Staking"],
            "influencer_sentiment": 0.7,
            "retail_sentiment": 0.6,
            "timestamp": datetime.now()
        }
    }


async def run_demo():
    """运行演示"""
    logger.info("=" * 60)
    logger.info("MultiAgentOrchestrator 演示开始")
    logger.info("=" * 60)
    
    # 1. 配置工作流
    config = WorkflowConfig(
        max_parallel_agents=6,
        enable_checkpointing=True,
        checkpoint_interval=3,
        timeout_seconds=180,
        retry_failed_nodes=True,
        max_retries=2,
        aggregation_method="ensemble",  # 使用集成方法
        consensus_threshold=0.65,
        enable_monitoring=True
    )
    
    logger.info("工作流配置:")
    logger.info(f"  - 最大并行Agent数: {config.max_parallel_agents}")
    logger.info(f"  - 聚合方法: {config.aggregation_method}")
    logger.info(f"  - 共识阈值: {config.consensus_threshold}")
    logger.info(f"  - 超时时间: {config.timeout_seconds}秒")
    
    # 2. 创建编排器
    orchestrator = MultiAgentOrchestrator(config=config)
    
    # 3. 注册回调函数（可选）
    def node_started_callback(state, error=None):
        logger.info(f"节点开始执行: Session={state.get('session_id')}")
    
    def node_completed_callback(state, error=None):
        logger.info(f"节点执行完成: Confidence scores={state.get('confidence_scores', {})}")
    
    def error_callback(state, error=None):
        if error:
            logger.error(f"节点执行错误: {error}")
    
    # 注册回调
    orchestrator.register_callback("before_data_preprocessing", node_started_callback)
    orchestrator.register_callback("after_parallel_analysis", node_completed_callback)
    orchestrator.register_callback("error_risk_assessment", error_callback)
    
    # 4. 准备初始状态
    initial_state = AgentState(
        session_id=f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        agent_id="demo_orchestrator",
        state_version=1,
        market_data=create_sample_market_data(),
        news_data=create_sample_news_data(),
        social_sentiment=create_sample_social_sentiment(),
        analyst_opinions=[],
        confidence_scores={},
        risk_assessment=None,
        portfolio_recommendations=[],
        final_decision=None,
        reasoning_chain=[],
        created_at=datetime.now(),
        updated_at=datetime.now(),
        metadata={
            "market_context": {
                "trend": "bullish",
                "volatility": "moderate",
                "macro_sentiment": "positive"
            }
        }
    )
    
    logger.info("\n初始状态准备完成:")
    logger.info(f"  - 市场数据: {len(initial_state['market_data'])} 个交易对")
    logger.info(f"  - 新闻数据: {len(initial_state['news_data'])} 条")
    logger.info(f"  - 社交情绪: {len(initial_state['social_sentiment'])} 个来源")
    
    # 5. 执行工作流
    logger.info("\n" + "=" * 60)
    logger.info("开始执行工作流...")
    logger.info("=" * 60)
    
    try:
        result = await orchestrator.execute_workflow(initial_state)
        
        if result['success']:
            logger.info("\n" + "=" * 60)
            logger.info("工作流执行成功!")
            logger.info("=" * 60)
            
            # 6. 展示结果
            logger.info("\n执行结果:")
            logger.info(f"  - 工作流ID: {result['workflow_id']}")
            
            # 展示指标
            metrics = result['metrics']
            logger.info(f"\n执行指标:")
            logger.info(f"  - 状态: {metrics['status']}")
            logger.info(f"  - 执行节点数: {metrics['nodes_executed']}")
            logger.info(f"  - 成功节点数: {metrics['nodes_successful']}")
            logger.info(f"  - 失败节点数: {metrics['nodes_failed']}")
            logger.info(f"  - 成功率: {metrics['success_rate']:.2%}")
            logger.info(f"  - 总执行时间: {metrics['total_execution_time_ms']:.2f}ms")
            
            # 展示决策
            decision = result.get('decision')
            if decision:
                logger.info(f"\n最终决策:")
                logger.info(f"  - 决策ID: {decision.get('decision_id')}")
                logger.info(f"  - 动作: {decision.get('action')}")
                logger.info(f"  - 总体置信度: {decision.get('total_confidence', 0):.2%}")
                logger.info(f"  - 风险调整分数: {decision.get('risk_adjusted_score', 0):.2f}")
                logger.info(f"  - 执行策略: {decision.get('execution_strategy')}")
                
                # 展示建议
                recommendations = decision.get('recommendations', [])
                if recommendations:
                    logger.info(f"\n投资建议 ({len(recommendations)}个):")
                    for i, rec in enumerate(recommendations[:5], 1):  # 只显示前5个
                        logger.info(f"  {i}. {rec['symbol']}")
                        logger.info(f"     - 动作: {rec['action']}")
                        logger.info(f"     - 仓位大小: {rec['position_size']:.2%}")
                        logger.info(f"     - 置信度: {rec['confidence']:.2%}")
            
            # 展示执行摘要
            summary = result.get('execution_summary', {})
            if summary:
                logger.info(f"\n节点执行详情:")
                for detail in summary.get('node_details', []):
                    status = "✓" if detail['success'] else "✗"
                    logger.info(f"  {status} {detail['node']}: {detail['time_ms']:.2f}ms")
                    if detail.get('error'):
                        logger.info(f"     错误: {detail['error']}")
            
            # 展示最终状态的关键信息
            final_state = result.get('final_state', {})
            if final_state:
                logger.info(f"\n分析师共识:")
                confidence_scores = final_state.get('confidence_scores', {})
                for analyst, confidence in confidence_scores.items():
                    logger.info(f"  - {analyst}: {confidence:.2%}")
                
                # 展示聚合结果
                aggregation_result = final_state.get('metadata', {}).get('aggregation_result', {})
                if aggregation_result:
                    logger.info(f"\n聚合分析:")
                    logger.info(f"  - 共识动作: {aggregation_result.get('consensus_action')}")
                    logger.info(f"  - 共识置信度: {aggregation_result.get('consensus_confidence', 0):.2%}")
                    logger.info(f"  - 一致性水平: {aggregation_result.get('agreement_level', 0):.2%}")
                    
                    # 关键洞察
                    insights = aggregation_result.get('key_insights', [])
                    if insights:
                        logger.info(f"\n关键洞察:")
                        for insight in insights[:3]:
                            logger.info(f"  • {insight}")
                    
                    # 风险因素
                    risks = aggregation_result.get('risk_factors', [])
                    if risks:
                        logger.info(f"\n主要风险:")
                        for risk in risks[:3]:
                            logger.info(f"  ⚠ {risk}")
        
        else:
            logger.error("\n" + "=" * 60)
            logger.error("工作流执行失败!")
            logger.error("=" * 60)
            logger.error(f"错误信息: {result.get('error')}")
            
            # 展示错误详情
            metrics = result.get('metrics', {})
            if metrics.get('error_messages'):
                logger.error("\n错误详情:")
                for error in metrics['error_messages']:
                    logger.error(f"  - {error}")
    
    except Exception as e:
        logger.error(f"\n执行异常: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    # 7. 获取执行历史
    logger.info("\n" + "=" * 60)
    logger.info("执行历史:")
    logger.info("=" * 60)
    
    history = orchestrator.get_execution_history(limit=5)
    for record in history:
        logger.info(f"  - {record['workflow_id']}: {record['status']} "
                   f"(成功率: {record['success_rate']:.2%}, "
                   f"耗时: {record['total_execution_time_ms']:.2f}ms)")
    
    # 8. 获取聚合器统计
    aggregator_stats = orchestrator.result_aggregator.get_aggregation_stats()
    logger.info("\n" + "=" * 60)
    logger.info("聚合器配置:")
    logger.info("=" * 60)
    logger.info(f"  - 方法: {aggregator_stats['method']}")
    logger.info(f"  - 共识阈值: {aggregator_stats['consensus_threshold']}")
    logger.info(f"  - 最小一致性: {aggregator_stats['min_agreement_level']}")
    
    logger.info("\n" + "=" * 60)
    logger.info("演示完成!")
    logger.info("=" * 60)


def main():
    """主函数"""
    # 运行异步演示
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()