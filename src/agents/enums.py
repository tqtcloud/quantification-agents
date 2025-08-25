"""
Agent相关枚举定义
"""

from enum import Enum


class InvestmentStyle(Enum):
    """投资风格枚举"""
    VALUE = "value"  # 价值投资
    GROWTH = "growth"  # 成长投资
    MOMENTUM = "momentum"  # 动量交易
    CONTRARIAN = "contrarian"  # 逆向投资
    TECHNICAL = "technical"  # 技术分析
    FUNDAMENTAL = "fundamental"  # 基本面分析
    MACRO = "macro"  # 宏观策略
    QUANTITATIVE = "quantitative"  # 量化策略
    ARBITRAGE = "arbitrage"  # 套利策略
    EVENT_DRIVEN = "event_driven"  # 事件驱动


class AnalysisType(Enum):
    """分析类型枚举"""
    MARKET_TREND = "market_trend"  # 市场趋势
    VALUATION = "valuation"  # 估值分析
    RISK_ASSESSMENT = "risk_assessment"  # 风险评估
    OPPORTUNITY = "opportunity"  # 机会识别
    PORTFOLIO = "portfolio"  # 组合建议
    TIMING = "timing"  # 时机判断