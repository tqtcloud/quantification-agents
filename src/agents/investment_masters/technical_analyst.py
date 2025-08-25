"""
技术分析专家Agent
专注于图表分析、技术指标和价格模式识别
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
import numpy as np

from src.agents.base_agent import InvestmentMasterAgent, InvestmentMasterConfig, MasterInsight
from src.agents.enums import InvestmentStyle, AnalysisType
from src.agents.models import FinalDecision, ReasoningStep
from src.core.models import TradingState, Signal


class TechnicalAnalystAgent(InvestmentMasterAgent):
    """技术分析专家Agent"""
    
    def __init__(self, config: Optional[InvestmentMasterConfig] = None, message_bus=None):
        """初始化技术分析专家Agent"""
        if config is None:
            config = InvestmentMasterConfig(
                name="technical_analyst_agent",
                master_name="Technical Analyst",
                investment_style=InvestmentStyle.TECHNICAL,
                specialty=[
                    "图表模式识别",
                    "技术指标分析",
                    "支撑阻力位",
                    "趋势跟踪",
                    "量价分析",
                    "市场时机"
                ],
                llm_model="gpt-4",
                llm_temperature=0.5,
                analysis_depth="comprehensive",
                risk_tolerance="moderate",
                time_horizon="short",  # 技术分析偏短期
                personality_traits={
                    "pattern_recognition": "expert",
                    "discipline": "high",
                    "timing_focus": "extreme",
                    "fundamental_ignore": "high",
                    "trend_following": "core",
                    "risk_management": "strict"
                },
                favorite_indicators=[
                    "MA_20", "MA_50", "MA_200",  # 移动平均线
                    "RSI", "MACD", "Stochastic",  # 动量指标
                    "Bollinger_Bands", "ATR",  # 波动性指标
                    "Volume", "OBV", "ADL",  # 成交量指标
                    "Support_Resistance", "Fibonacci"  # 价格水平
                ],
                avoid_sectors=[]  # 技术分析不偏向特定行业
            )
        
        super().__init__(config, message_bus)
        
        # 技术分析参数
        self.technical_params = {
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "macd_signal_threshold": 0.02,
            "volume_spike_multiplier": 1.5,
            "breakout_confirmation_bars": 2,
            "support_resistance_tolerance": 0.02,  # 2%容差
            "trend_strength_minimum": 0.6
        }
        
        # 图表模式库
        self.chart_patterns = {
            "head_and_shoulders": {"reliability": 0.85, "type": "reversal"},
            "double_top": {"reliability": 0.75, "type": "reversal"},
            "double_bottom": {"reliability": 0.75, "type": "reversal"},
            "ascending_triangle": {"reliability": 0.70, "type": "continuation"},
            "descending_triangle": {"reliability": 0.70, "type": "continuation"},
            "flag": {"reliability": 0.65, "type": "continuation"},
            "wedge": {"reliability": 0.60, "type": "reversal"},
            "cup_and_handle": {"reliability": 0.80, "type": "continuation"}
        }
    
    async def make_investment_decision(self, 
                                      state: TradingState,
                                      portfolio: Dict[str, Any]) -> FinalDecision:
        """
        做出投资决策
        技术分析风格：基于价格行为和技术指标
        """
        reasoning_steps = []
        
        # 步骤1：趋势分析
        trend_step = await self._analyze_trend(state)
        reasoning_steps.append(trend_step)
        
        # 步骤2：动量分析
        momentum_step = await self._analyze_momentum(state)
        reasoning_steps.append(momentum_step)
        
        # 步骤3：支撑阻力分析
        support_resistance_step = await self._analyze_support_resistance(state)
        reasoning_steps.append(support_resistance_step)
        
        # 步骤4：图表模式识别
        pattern_step = await self._identify_chart_patterns(state)
        reasoning_steps.append(pattern_step)
        
        # 步骤5：成交量分析
        volume_step = await self._analyze_volume(state)
        reasoning_steps.append(volume_step)
        
        # 综合决策
        decision = self._synthesize_decision(reasoning_steps, portfolio)
        
        return FinalDecision(
            action=decision["action"],
            confidence=decision["confidence"],
            reasoning_chain=reasoning_steps,
            position_size=decision["position_size"],
            risk_assessment=decision["risk_assessment"],
            expected_return=decision["expected_return"],
            time_horizon=decision["time_horizon"],
            metadata={
                "investment_style": "Technical Analysis",
                "primary_signal": decision["primary_signal"],
                "stop_loss": decision["stop_loss"],
                "take_profit": decision["take_profit"],
                "entry_point": decision["entry_point"]
            }
        )
    
    async def _analyze_trend(self, state: TradingState) -> ReasoningStep:
        """分析趋势"""
        trend_analysis = {}
        
        for symbol in state.active_symbols:
            if symbol in state.market_data:
                data = state.market_data[symbol]
                current_price = data.close
                
                # 简化的趋势判断
                trends = {
                    "short_term": self._calculate_trend_strength(current_price, 20),
                    "medium_term": self._calculate_trend_strength(current_price, 50),
                    "long_term": self._calculate_trend_strength(current_price, 200)
                }
                
                # 综合趋势方向
                trend_score = np.mean(list(trends.values()))
                if trend_score > 0.3:
                    direction = "uptrend"
                elif trend_score < -0.3:
                    direction = "downtrend"
                else:
                    direction = "sideways"
                
                trend_analysis[symbol] = {
                    "direction": direction,
                    "strength": abs(trend_score),
                    "trends": trends,
                    "tradeable": abs(trend_score) > 0.3
                }
        
        avg_strength = np.mean([t["strength"] for t in trend_analysis.values()])
        
        return ReasoningStep(
            thought="趋势分析 - 趋势是你的朋友",
            action="trend_analysis",
            observation=f"平均趋势强度: {avg_strength:.2f}",
            confidence=avg_strength,
            metadata={"trend_analysis": trend_analysis}
        )
    
    async def _analyze_momentum(self, state: TradingState) -> ReasoningStep:
        """分析动量指标"""
        momentum_signals = {}
        
        for symbol in state.active_symbols:
            signals = {
                "rsi": self._calculate_rsi_signal(symbol, state),
                "macd": self._calculate_macd_signal(symbol, state),
                "stochastic": self._calculate_stochastic_signal(symbol, state)
            }
            
            # 综合动量信号
            bullish_count = sum(1 for s in signals.values() if s > 0)
            bearish_count = sum(1 for s in signals.values() if s < 0)
            
            if bullish_count > bearish_count:
                momentum = "bullish"
                strength = bullish_count / len(signals)
            elif bearish_count > bullish_count:
                momentum = "bearish"
                strength = bearish_count / len(signals)
            else:
                momentum = "neutral"
                strength = 0
            
            momentum_signals[symbol] = {
                "momentum": momentum,
                "strength": strength,
                "signals": signals
            }
        
        avg_momentum = np.mean([m["strength"] for m in momentum_signals.values()])
        
        return ReasoningStep(
            thought="动量分析 - 识别超买超卖和动量转折",
            action="momentum_analysis",
            observation=f"平均动量强度: {avg_momentum:.2f}",
            confidence=avg_momentum * 0.8,
            metadata={"momentum_signals": momentum_signals}
        )
    
    async def _analyze_support_resistance(self, state: TradingState) -> ReasoningStep:
        """分析支撑阻力位"""
        sr_analysis = {}
        
        for symbol in state.active_symbols:
            if symbol in state.market_data:
                data = state.market_data[symbol]
                current_price = data.close
                
                # 简化的支撑阻力计算
                support_levels = [
                    current_price * 0.95,  # 5%下方
                    current_price * 0.90,  # 10%下方
                    data.low  # 日内低点
                ]
                
                resistance_levels = [
                    current_price * 1.05,  # 5%上方
                    current_price * 1.10,  # 10%上方
                    data.high  # 日内高点
                ]
                
                # 计算到最近支撑/阻力的距离
                nearest_support = max([s for s in support_levels if s < current_price], default=0)
                nearest_resistance = min([r for r in resistance_levels if r > current_price], default=float('inf'))
                
                position_in_range = (current_price - nearest_support) / (nearest_resistance - nearest_support) if nearest_resistance != float('inf') else 0.5
                
                sr_analysis[symbol] = {
                    "support": nearest_support,
                    "resistance": nearest_resistance,
                    "position": position_in_range,
                    "near_support": position_in_range < 0.3,
                    "near_resistance": position_in_range > 0.7
                }
        
        support_opportunities = sum(1 for s in sr_analysis.values() if s["near_support"])
        resistance_warnings = sum(1 for s in sr_analysis.values() if s["near_resistance"])
        
        return ReasoningStep(
            thought="支撑阻力分析 - 关键价格水平的买卖点",
            action="support_resistance_analysis",
            observation=f"接近支撑: {support_opportunities}, 接近阻力: {resistance_warnings}",
            confidence=0.7,
            metadata={"sr_analysis": sr_analysis}
        )
    
    async def _identify_chart_patterns(self, state: TradingState) -> ReasoningStep:
        """识别图表模式"""
        pattern_detection = {}
        
        for symbol in state.active_symbols:
            detected_patterns = []
            
            # 简化的模式识别（实际需要复杂的算法）
            if self._detect_pattern("double_bottom", symbol, state):
                detected_patterns.append({
                    "pattern": "double_bottom",
                    "reliability": self.chart_patterns["double_bottom"]["reliability"],
                    "type": "reversal",
                    "implication": "bullish"
                })
            
            if self._detect_pattern("head_and_shoulders", symbol, state):
                detected_patterns.append({
                    "pattern": "head_and_shoulders",
                    "reliability": self.chart_patterns["head_and_shoulders"]["reliability"],
                    "type": "reversal",
                    "implication": "bearish"
                })
            
            pattern_detection[symbol] = {
                "patterns": detected_patterns,
                "has_pattern": len(detected_patterns) > 0,
                "max_reliability": max([p["reliability"] for p in detected_patterns], default=0)
            }
        
        patterns_found = sum(1 for p in pattern_detection.values() if p["has_pattern"])
        avg_reliability = np.mean([p["max_reliability"] for p in pattern_detection.values()])
        
        return ReasoningStep(
            thought="图表模式识别 - 寻找经典的技术形态",
            action="pattern_recognition",
            observation=f"发现模式: {patterns_found}, 平均可靠性: {avg_reliability:.2f}",
            confidence=avg_reliability,
            metadata={"pattern_detection": pattern_detection}
        )
    
    async def _analyze_volume(self, state: TradingState) -> ReasoningStep:
        """成交量分析"""
        volume_analysis = {}
        
        for symbol in state.active_symbols:
            if symbol in state.market_data:
                data = state.market_data[symbol]
                current_volume = data.volume
                
                # 简化的成交量分析
                avg_volume = current_volume  # 实际应该用历史平均
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                
                # 量价配合分析
                price_change = (data.close - data.open) / data.open if data.open > 0 else 0
                
                if volume_ratio > self.technical_params["volume_spike_multiplier"]:
                    if price_change > 0:
                        signal = "bullish_volume"
                    else:
                        signal = "bearish_volume"
                elif volume_ratio < 0.5:
                    signal = "low_volume_warning"
                else:
                    signal = "normal_volume"
                
                volume_analysis[symbol] = {
                    "volume_ratio": volume_ratio,
                    "signal": signal,
                    "price_volume_confirm": (price_change > 0 and volume_ratio > 1) or (price_change < 0 and volume_ratio > 1)
                }
        
        confirmation_count = sum(1 for v in volume_analysis.values() if v["price_volume_confirm"])
        
        return ReasoningStep(
            thought="成交量分析 - 量价配合验证趋势",
            action="volume_analysis",
            observation=f"量价配合确认: {confirmation_count}/{len(volume_analysis)}",
            confidence=confirmation_count / max(len(volume_analysis), 1),
            metadata={"volume_analysis": volume_analysis}
        )
    
    def _synthesize_decision(self, reasoning_steps: List[ReasoningStep], 
                            portfolio: Dict[str, Any]) -> Dict[str, Any]:
        """综合所有技术分析做出决策"""
        # 提取各分析结果
        trend_data = reasoning_steps[0].metadata.get("trend_analysis", {})
        momentum_data = reasoning_steps[1].metadata.get("momentum_signals", {})
        sr_data = reasoning_steps[2].metadata.get("sr_analysis", {})
        pattern_data = reasoning_steps[3].metadata.get("pattern_detection", {})
        volume_data = reasoning_steps[4].metadata.get("volume_analysis", {})
        
        # 计算综合信号
        buy_signals = 0
        sell_signals = 0
        
        # 趋势信号
        for symbol, trend in trend_data.items():
            if trend["direction"] == "uptrend" and trend["strength"] > 0.5:
                buy_signals += 1
            elif trend["direction"] == "downtrend" and trend["strength"] > 0.5:
                sell_signals += 1
        
        # 动量信号
        for symbol, momentum in momentum_data.items():
            if momentum["momentum"] == "bullish":
                buy_signals += 1
            elif momentum["momentum"] == "bearish":
                sell_signals += 1
        
        # 支撑阻力信号
        for symbol, sr in sr_data.items():
            if sr["near_support"]:
                buy_signals += 1
            elif sr["near_resistance"]:
                sell_signals += 1
        
        # 决策逻辑
        total_signals = max(buy_signals + sell_signals, 1)
        buy_ratio = buy_signals / total_signals
        sell_ratio = sell_signals / total_signals
        
        if buy_ratio > 0.6:
            action = "BUY"
            confidence = buy_ratio
            position_size = min(confidence * 0.15, 0.10)  # 最多10%仓位
            primary_signal = "Multiple buy signals aligned"
        elif sell_ratio > 0.6:
            action = "SELL"
            confidence = sell_ratio
            position_size = 0.0
            primary_signal = "Multiple sell signals aligned"
        else:
            action = "HOLD"
            confidence = 0.5
            position_size = 0.0
            primary_signal = "Mixed signals - wait for clarity"
        
        # 计算止损和止盈
        entry_point, stop_loss, take_profit = self._calculate_risk_levels(action, sr_data)
        
        return {
            "action": action,
            "confidence": confidence,
            "position_size": position_size,
            "risk_assessment": "Technical levels based risk management",
            "expected_return": "5-15% per trade",
            "time_horizon": "Days to weeks",
            "primary_signal": primary_signal,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "entry_point": entry_point
        }
    
    def _calculate_trend_strength(self, price: float, period: int) -> float:
        """计算趋势强度（简化）"""
        # 实际应该用移动平均线
        return np.random.uniform(-1, 1)  # 模拟值
    
    def _calculate_rsi_signal(self, symbol: str, state: TradingState) -> float:
        """计算RSI信号"""
        # 简化：返回模拟值
        rsi = np.random.uniform(20, 80)
        if rsi < self.technical_params["rsi_oversold"]:
            return 1  # 买入信号
        elif rsi > self.technical_params["rsi_overbought"]:
            return -1  # 卖出信号
        return 0
    
    def _calculate_macd_signal(self, symbol: str, state: TradingState) -> float:
        """计算MACD信号"""
        # 简化：返回模拟值
        return np.random.choice([-1, 0, 1])
    
    def _calculate_stochastic_signal(self, symbol: str, state: TradingState) -> float:
        """计算随机指标信号"""
        # 简化：返回模拟值
        return np.random.choice([-1, 0, 1])
    
    def _detect_pattern(self, pattern_name: str, symbol: str, state: TradingState) -> bool:
        """检测特定图表模式（简化）"""
        # 实际需要复杂的模式识别算法
        return np.random.random() < 0.2  # 20%概率检测到模式
    
    def _calculate_risk_levels(self, action: str, sr_data: Dict[str, Any]) -> tuple:
        """计算风险水平"""
        # 取第一个标的的支撑阻力（简化）
        if sr_data:
            first_symbol = list(sr_data.keys())[0]
            sr = sr_data[first_symbol]
            
            if action == "BUY":
                entry = sr["support"] * 1.01  # 支撑位上方进入
                stop_loss = sr["support"] * 0.98  # 支撑位下方止损
                take_profit = sr["resistance"] * 0.98  # 阻力位下方止盈
            elif action == "SELL":
                entry = sr["resistance"] * 0.99  # 阻力位下方进入
                stop_loss = sr["resistance"] * 1.02  # 阻力位上方止损
                take_profit = sr["support"] * 1.02  # 支撑位上方止盈
            else:
                entry = stop_loss = take_profit = 0
            
            return entry, stop_loss, take_profit
        
        return 0, 0, 0
    
    async def evaluate_portfolio(self, 
                                portfolio: Dict[str, Any],
                                market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估投资组合
        技术分析风格：关注趋势和时机
        """
        evaluation = {
            "trend_alignment": 0.0,
            "timing_quality": 0.0,
            "risk_reward_ratio": 0.0,
            "technical_health": 0.0,
            "recommendations": []
        }
        
        positions = portfolio.get("positions", {})
        
        # 评估趋势对齐
        aligned_count = 0
        for symbol, position in positions.items():
            # 简化：检查是否与趋势对齐
            if position.get("unrealized_return", 0) > 0:
                aligned_count += 1
        
        if len(positions) > 0:
            evaluation["trend_alignment"] = aligned_count / len(positions)
        
        # 评估进场时机
        evaluation["timing_quality"] = 0.7  # 简化评分
        
        # 评估风险回报比
        avg_rr_ratio = 2.5  # 假设平均风险回报比
        if avg_rr_ratio >= 3:
            evaluation["risk_reward_ratio"] = 0.9
        elif avg_rr_ratio >= 2:
            evaluation["risk_reward_ratio"] = 0.7
        else:
            evaluation["risk_reward_ratio"] = 0.5
            evaluation["recommendations"].append("改善风险回报比，目标至少2:1")
        
        # 技术健康度
        market_trend = market_conditions.get("trend", "neutral")
        if market_trend == "bullish":
            evaluation["technical_health"] = 0.8
        elif market_trend == "bearish":
            evaluation["technical_health"] = 0.4
            evaluation["recommendations"].append("市场趋势不利，考虑减少仓位")
        else:
            evaluation["technical_health"] = 0.6
        
        # 总体评分
        evaluation["overall_score"] = (
            evaluation["trend_alignment"] * 0.30 +
            evaluation["timing_quality"] * 0.25 +
            evaluation["risk_reward_ratio"] * 0.25 +
            evaluation["technical_health"] * 0.20
        )
        
        # 技术分析建议
        evaluation["technical_advice"] = self._get_technical_advice(evaluation["overall_score"])
        
        return evaluation
    
    def _get_technical_advice(self, score: float) -> str:
        """获取技术分析建议"""
        if score >= 0.8:
            return "技术面强劲，继续跟随趋势。记住：让利润奔跑，快速止损。"
        elif score >= 0.6:
            return "技术面中性，保持纪律。价格包含一切，专注于图表告诉你的信息。"
        elif score >= 0.4:
            return "技术面转弱，提高警惕。当趋势不明确时，空仓也是一种仓位。"
        else:
            return "技术面疲弱，考虑减仓。保护资本是第一要务，机会永远存在。"