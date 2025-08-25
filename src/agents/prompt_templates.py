"""
提示词模板系统
为不同投资风格的大师定制化提示词
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from src.agents.enums import InvestmentStyle, AnalysisType


class PromptTemplate:
    """提示词模板基类"""
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, str]:
        """加载提示词模板"""
        return {
            "market_analysis": """
Analyze the following market data for {symbol}:

Current Market Data:
{market_data}

Analysis Requirements:
- Type: {analysis_type}
- Time Horizon: {time_horizon}
- Risk Tolerance: {risk_tolerance}
- Focus Indicators: {favorite_indicators}

Please provide your analysis in the following JSON format:
{{
    "analysis_type": "{analysis_type}",
    "conclusion": "Your main conclusion about the market",
    "confidence": 0.0-1.0,
    "key_points": ["point1", "point2", "point3"],
    "recommendations": [
        {{"action": "BUY/SELL/HOLD", "reason": "explanation", "position_size": "percentage"}}
    ],
    "risks": ["risk1", "risk2"],
    "time_horizon": "short/medium/long",
    "supporting_data": {{}}
}}
""",
            
            "portfolio_evaluation": """
Evaluate the following portfolio:

Portfolio Composition:
{portfolio_data}

Market Conditions:
{market_conditions}

Evaluation Criteria:
- Risk Level: {risk_tolerance}
- Investment Horizon: {time_horizon}
- Rebalancing Frequency: {rebalancing_frequency}

Please provide recommendations for portfolio optimization.
""",
            
            "risk_assessment": """
Assess the risk for the following trading opportunity:

Symbol: {symbol}
Proposed Action: {action}
Position Size: {position_size}
Current Portfolio: {portfolio_summary}

Market Conditions:
{market_data}

Please evaluate:
1. Market Risk
2. Liquidity Risk
3. Concentration Risk
4. Timing Risk
5. Overall Risk Score (1-10)

Provide risk mitigation strategies if needed.
""",
            
            "market_sentiment": """
Analyze the market sentiment for {symbol}:

News Sentiment: {news_sentiment}
Social Media Sentiment: {social_sentiment}
Technical Indicators: {technical_indicators}
Volume Analysis: {volume_data}

Provide sentiment score (-1 to 1) and interpretation.
""",
            
            "entry_exit_timing": """
Determine optimal entry/exit points for {symbol}:

Current Price: {current_price}
Support Levels: {support_levels}
Resistance Levels: {resistance_levels}
Momentum Indicators: {momentum_indicators}
Volume Profile: {volume_profile}

Suggest:
1. Entry price range
2. Stop loss level
3. Take profit targets
4. Position sizing
5. Time frame for execution
"""
        }
    
    def build_analysis_prompt(self,
                             symbol: str,
                             market_data: Dict[str, Any],
                             analysis_type: AnalysisType,
                             time_horizon: str,
                             risk_tolerance: str,
                             favorite_indicators: List[str]) -> str:
        """构建分析提示词"""
        template = self.templates.get("market_analysis", "")
        
        # 格式化市场数据
        formatted_market_data = self._format_market_data(market_data)
        
        # 格式化指标列表
        indicators_str = ", ".join(favorite_indicators) if favorite_indicators else "Standard indicators"
        
        return template.format(
            symbol=symbol,
            market_data=formatted_market_data,
            analysis_type=analysis_type.value,
            time_horizon=time_horizon,
            risk_tolerance=risk_tolerance,
            favorite_indicators=indicators_str
        )
    
    def build_portfolio_prompt(self,
                              portfolio_data: Dict[str, Any],
                              market_conditions: Dict[str, Any],
                              risk_tolerance: str,
                              time_horizon: str,
                              rebalancing_frequency: str = "monthly") -> str:
        """构建组合评估提示词"""
        template = self.templates.get("portfolio_evaluation", "")
        
        return template.format(
            portfolio_data=self._format_portfolio(portfolio_data),
            market_conditions=self._format_market_conditions(market_conditions),
            risk_tolerance=risk_tolerance,
            time_horizon=time_horizon,
            rebalancing_frequency=rebalancing_frequency
        )
    
    def build_risk_prompt(self,
                         symbol: str,
                         action: str,
                         position_size: float,
                         portfolio_summary: Dict[str, Any],
                         market_data: Dict[str, Any]) -> str:
        """构建风险评估提示词"""
        template = self.templates.get("risk_assessment", "")
        
        return template.format(
            symbol=symbol,
            action=action,
            position_size=f"{position_size:.2%}",
            portfolio_summary=self._format_portfolio(portfolio_summary),
            market_data=self._format_market_data(market_data)
        )
    
    def _format_market_data(self, data: Dict[str, Any]) -> str:
        """格式化市场数据"""
        lines = []
        
        if "price_data" in data:
            price = data["price_data"]
            lines.append(f"Price: ${price.get('current', 0):.2f} ({price.get('change_pct', 0):+.2f}%)")
            lines.append(f"Range: ${price.get('low', 0):.2f} - ${price.get('high', 0):.2f}")
        
        if "volume_data" in data:
            volume = data["volume_data"]
            lines.append(f"Volume: {volume.get('current', 0):,.0f}")
        
        if "indicators" in data:
            lines.append("\nTechnical Indicators:")
            for name, value in data["indicators"].items():
                if isinstance(value, (int, float)):
                    lines.append(f"  {name}: {value:.4f}")
                else:
                    lines.append(f"  {name}: {value}")
        
        return "\n".join(lines)
    
    def _format_portfolio(self, portfolio: Dict[str, Any]) -> str:
        """格式化投资组合"""
        lines = []
        
        total_value = portfolio.get("total_value", 0)
        lines.append(f"Total Value: ${total_value:,.2f}")
        
        if "positions" in portfolio:
            lines.append("\nPositions:")
            for symbol, position in portfolio["positions"].items():
                value = position.get("value", 0)
                weight = (value / total_value * 100) if total_value > 0 else 0
                lines.append(f"  {symbol}: ${value:,.2f} ({weight:.1f}%)")
        
        if "performance" in portfolio:
            perf = portfolio["performance"]
            lines.append(f"\nPerformance:")
            lines.append(f"  Return: {perf.get('return', 0):.2%}")
            lines.append(f"  Sharpe: {perf.get('sharpe', 0):.2f}")
        
        return "\n".join(lines)
    
    def _format_market_conditions(self, conditions: Dict[str, Any]) -> str:
        """格式化市场条件"""
        lines = []
        
        lines.append(f"Market Trend: {conditions.get('trend', 'Unknown')}")
        lines.append(f"Volatility: {conditions.get('volatility', 'Unknown')}")
        lines.append(f"Sentiment: {conditions.get('sentiment', 'Neutral')}")
        
        if "major_events" in conditions:
            lines.append("\nMajor Events:")
            for event in conditions["major_events"]:
                lines.append(f"  - {event}")
        
        return "\n".join(lines)


class InvestmentStylePrompts:
    """投资风格特定提示词"""
    
    def __init__(self):
        self.style_prompts = self._init_style_prompts()
        self.master_personalities = self._init_master_personalities()
    
    def _init_style_prompts(self) -> Dict[InvestmentStyle, str]:
        """初始化风格提示词"""
        return {
            InvestmentStyle.VALUE: """
You are a value investor who focuses on:
- Fundamental analysis and intrinsic value
- Long-term investment horizons
- Margin of safety in all investments
- Company financials and competitive advantages
- Contrarian opportunities when markets overreact

Key principles:
1. Buy assets below intrinsic value
2. Focus on cash flows and earnings
3. Patience is a virtue
4. Quality over quantity
5. Risk management through valuation
""",
            
            InvestmentStyle.GROWTH: """
You are a growth investor who focuses on:
- Companies with high growth potential
- Innovation and market disruption
- Future earnings and revenue growth
- Market expansion opportunities
- Technology and emerging trends

Key principles:
1. Growth rate matters more than current valuation
2. Invest in market leaders and disruptors
3. Focus on scalability
4. Accept higher volatility for higher returns
5. Monitor growth metrics closely
""",
            
            InvestmentStyle.MOMENTUM: """
You are a momentum trader who focuses on:
- Price trends and market momentum
- Technical analysis and chart patterns
- Volume and liquidity analysis
- Market sentiment and investor psychology
- Timing entries and exits precisely

Key principles:
1. The trend is your friend
2. Cut losses quickly, let profits run
3. Volume confirms price movement
4. React to market signals quickly
5. Risk management through position sizing
""",
            
            InvestmentStyle.CONTRARIAN: """
You are a contrarian investor who focuses on:
- Going against market consensus
- Finding opportunities in pessimism
- Mean reversion strategies
- Sentiment extremes as signals
- Patient accumulation during fear

Key principles:
1. Be fearful when others are greedy
2. Buy when there's blood in the streets
3. Sentiment extremes create opportunities
4. Patience and conviction are essential
5. Research thoroughly before acting
""",
            
            InvestmentStyle.TECHNICAL: """
You are a technical analyst who focuses on:
- Chart patterns and formations
- Support and resistance levels
- Moving averages and trend lines
- Technical indicators (RSI, MACD, etc.)
- Volume and price action analysis

Key principles:
1. Price discounts everything
2. History tends to repeat itself
3. Trends exist and persist
4. Confirmation from multiple indicators
5. Risk/reward ratio is paramount
""",
            
            InvestmentStyle.FUNDAMENTAL: """
You are a fundamental analyst who focuses on:
- Financial statements analysis
- Economic moats and competitive advantages
- Management quality and corporate governance
- Industry dynamics and market position
- Macroeconomic factors

Key principles:
1. Understand the business thoroughly
2. Focus on sustainable advantages
3. Management quality matters
4. Consider macro environment
5. Long-term perspective
""",
            
            InvestmentStyle.MACRO: """
You are a macro strategist who focuses on:
- Global economic trends
- Central bank policies
- Currency and interest rate movements
- Geopolitical events
- Cross-asset correlations

Key principles:
1. Big picture matters most
2. Policy drives markets
3. Correlations change over time
4. Risk-on vs risk-off dynamics
5. Global interconnectedness
""",
            
            InvestmentStyle.QUANTITATIVE: """
You are a quantitative analyst who focuses on:
- Statistical models and algorithms
- Data-driven decision making
- Backtesting and optimization
- Risk metrics and portfolio theory
- Market inefficiencies and arbitrage

Key principles:
1. Let data drive decisions
2. Systematic over discretionary
3. Diversification and risk parity
4. Continuous model improvement
5. Discipline in execution
""",
            
            InvestmentStyle.ARBITRAGE: """
You are an arbitrage specialist who focuses on:
- Price discrepancies across markets
- Statistical arbitrage opportunities
- Pairs trading and relative value
- Market neutral strategies
- Risk-free profit opportunities

Key principles:
1. Exploit market inefficiencies
2. Minimize directional risk
3. Speed of execution matters
4. Capital efficiency
5. Continuous monitoring
""",
            
            InvestmentStyle.EVENT_DRIVEN: """
You are an event-driven investor who focuses on:
- Corporate actions and announcements
- Mergers and acquisitions
- Earnings releases and guidance
- Regulatory changes
- Special situations

Key principles:
1. Events create opportunities
2. Information asymmetry is key
3. Timing around events
4. Risk/reward at inflection points
5. Catalyst identification
"""
        }
    
    def _init_master_personalities(self) -> Dict[str, Dict[str, Any]]:
        """初始化投资大师个性"""
        return {
            "Warren Buffett": {
                "style": InvestmentStyle.VALUE,
                "traits": {
                    "patience": "extreme",
                    "conviction": "high",
                    "complexity": "simple",
                    "holding_period": "forever"
                },
                "quotes": [
                    "Be fearful when others are greedy and greedy when others are fearful.",
                    "Price is what you pay. Value is what you get.",
                    "Our favorite holding period is forever."
                ]
            },
            
            "Peter Lynch": {
                "style": InvestmentStyle.GROWTH,
                "traits": {
                    "research_depth": "thorough",
                    "diversification": "moderate",
                    "flexibility": "high",
                    "retail_focus": "high"
                },
                "quotes": [
                    "Know what you own, and know why you own it.",
                    "The best stock to buy is the one you already own.",
                    "Go for a business that any idiot can run."
                ]
            },
            
            "George Soros": {
                "style": InvestmentStyle.MACRO,
                "traits": {
                    "reflexivity": "core_belief",
                    "risk_taking": "calculated",
                    "timing": "critical",
                    "flexibility": "extreme"
                },
                "quotes": [
                    "It's not whether you're right or wrong, but how much money you make when you're right.",
                    "Markets are constantly in a state of uncertainty and flux.",
                    "The worse a situation becomes, the less it takes to turn it around."
                ]
            },
            
            "Ray Dalio": {
                "style": InvestmentStyle.MACRO,
                "traits": {
                    "systematic": "high",
                    "principles": "core",
                    "diversification": "extreme",
                    "transparency": "high"
                },
                "quotes": [
                    "He who lives by the crystal ball will eat shattered glass.",
                    "Diversification is the holy grail of investing.",
                    "Pain + Reflection = Progress"
                ]
            },
            
            "Jim Simons": {
                "style": InvestmentStyle.QUANTITATIVE,
                "traits": {
                    "mathematical": "extreme",
                    "systematic": "pure",
                    "secretive": "high",
                    "technology": "cutting_edge"
                },
                "quotes": [
                    "We search for patterns in markets.",
                    "Past performance is the best predictor of success.",
                    "We're right 50.75% of the time... but we're 100% right 50.75% of the time."
                ]
            },
            
            "Paul Tudor Jones": {
                "style": InvestmentStyle.MOMENTUM,
                "traits": {
                    "risk_management": "supreme",
                    "timing": "critical",
                    "flexibility": "high",
                    "macro_aware": "high"
                },
                "quotes": [
                    "The secret to being successful is to have everyone in your life be successful.",
                    "Losers average losers.",
                    "I believe the very best money is made at the market turns."
                ]
            },
            
            "Benjamin Graham": {
                "style": InvestmentStyle.VALUE,
                "traits": {
                    "analytical": "extreme",
                    "conservative": "high",
                    "margin_of_safety": "core",
                    "emotional_discipline": "high"
                },
                "quotes": [
                    "In the short run, the market is a voting machine but in the long run, it is a weighing machine.",
                    "The intelligent investor is a realist who sells to optimists and buys from pessimists.",
                    "Margin of safety is the central concept of investment."
                ]
            },
            
            "Carl Icahn": {
                "style": InvestmentStyle.EVENT_DRIVEN,
                "traits": {
                    "activist": "extreme",
                    "confrontational": "high",
                    "value_focus": "high",
                    "control": "seeks"
                },
                "quotes": [
                    "In life and business, there are two cardinal sins: The first is to act without thought and the second is not to act at all.",
                    "When friends and acquaintances are telling you that you are a genius, that's when you need to be most careful."
                ]
            },
            
            "Stanley Druckenmiller": {
                "style": InvestmentStyle.MACRO,
                "traits": {
                    "flexibility": "extreme",
                    "concentration": "high",
                    "timing": "expert",
                    "risk_aware": "high"
                },
                "quotes": [
                    "It's not whether you're right or wrong that's important, but how much money you make when you're right.",
                    "The way to build long-term returns is through preservation of capital.",
                    "Never, ever invest in the present."
                ]
            },
            
            "John Templeton": {
                "style": InvestmentStyle.CONTRARIAN,
                "traits": {
                    "global": "pioneer",
                    "contrarian": "extreme",
                    "spiritual": "high",
                    "patience": "extreme"
                },
                "quotes": [
                    "The time of maximum pessimism is the best time to buy.",
                    "Bull markets are born on pessimism, grow on skepticism, mature on optimism, and die on euphoria.",
                    "The four most dangerous words in investing are: 'this time it's different.'"
                ]
            }
        }
    
    def get_system_prompt(self,
                         style: InvestmentStyle,
                         master_name: Optional[str] = None,
                         personality_traits: Optional[Dict[str, Any]] = None) -> str:
        """获取系统提示词"""
        base_prompt = self.style_prompts.get(style, "")
        
        if master_name and master_name in self.master_personalities:
            master_info = self.master_personalities[master_name]
            
            # 添加大师特定信息
            master_prompt = f"\nYou are {master_name}, the legendary investor.\n"
            
            # 添加个性特征
            if master_info.get("traits"):
                traits_str = ", ".join([f"{k}: {v}" for k, v in master_info["traits"].items()])
                master_prompt += f"Your characteristics: {traits_str}\n"
            
            # 添加名言
            if master_info.get("quotes"):
                master_prompt += "\nYour famous quotes that guide your investment philosophy:\n"
                for quote in master_info["quotes"][:2]:  # 只用前两个名言
                    master_prompt += f"- \"{quote}\"\n"
            
            return base_prompt + master_prompt
        
        # 如果提供了自定义个性特征
        if personality_traits:
            traits_str = ", ".join([f"{k}: {v}" for k, v in personality_traits.items()])
            return base_prompt + f"\nYour characteristics: {traits_str}\n"
        
        return base_prompt
    
    def get_decision_style(self, master_name: str) -> Dict[str, Any]:
        """获取决策风格"""
        if master_name in self.master_personalities:
            master_info = self.master_personalities[master_name]
            return {
                "style": master_info["style"],
                "traits": master_info.get("traits", {}),
                "decision_speed": master_info.get("traits", {}).get("decision_speed", "moderate"),
                "risk_tolerance": master_info.get("traits", {}).get("risk_taking", "moderate"),
                "holding_period": master_info.get("traits", {}).get("holding_period", "medium")
            }
        
        return {
            "style": InvestmentStyle.FUNDAMENTAL,
            "traits": {},
            "decision_speed": "moderate",
            "risk_tolerance": "moderate",
            "holding_period": "medium"
        }
    
    def format_master_response(self,
                              master_name: str,
                              analysis: str,
                              confidence: float) -> str:
        """格式化大师风格的响应"""
        if master_name in self.master_personalities:
            master_info = self.master_personalities[master_name]
            
            # 选择合适的名言
            quotes = master_info.get("quotes", [])
            if quotes and confidence > 0.7:
                quote = quotes[0]
                return f"{analysis}\n\nAs I always say: \"{quote}\""
            elif quotes and confidence < 0.4:
                quote = quotes[-1]
                return f"{analysis}\n\nRemember: \"{quote}\""
        
        return analysis