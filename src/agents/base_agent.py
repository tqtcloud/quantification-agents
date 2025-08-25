"""
投资大师Agent基类
实现投资大师的分析框架和LLM调用能力
"""

import asyncio
import json
import time
from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from src.agents.base import BaseAgent, AgentConfig
from src.agents.enums import InvestmentStyle, AnalysisType
from src.agents.llm_client import LLMClient, LLMConfig, LLMResponse
from src.agents.models import AgentState, ReasoningStep, FinalDecision
from src.agents.prompt_templates import PromptTemplate, InvestmentStylePrompts
from src.core.models import MarketData, Signal, TradingState
from src.utils.logger import LoggerMixin


@dataclass
class MasterInsight:
    """投资大师洞察"""
    master_name: str
    investment_style: InvestmentStyle
    analysis_type: AnalysisType
    main_conclusion: str
    confidence_score: float
    key_points: List[str]
    recommendations: List[Dict[str, Any]]
    risk_warnings: List[str]
    time_horizon: str  # 短期/中期/长期
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class InvestmentMasterConfig(AgentConfig):
    """投资大师Agent配置"""
    master_name: str = Field(description="投资大师名称")
    investment_style: InvestmentStyle = Field(description="投资风格")
    specialty: List[str] = Field(default_factory=list, description="专业领域")
    
    # LLM配置
    llm_provider: str = Field(default="openai", description="LLM提供商")
    llm_model: str = Field(default="gpt-4", description="模型名称")
    llm_temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    llm_max_tokens: int = Field(default=2000, ge=100, le=8000, description="最大tokens")
    llm_timeout: float = Field(default=30.0, description="超时时间(秒)")
    
    # 分析配置
    analysis_depth: str = Field(default="comprehensive", description="分析深度：quick/standard/comprehensive")
    risk_tolerance: str = Field(default="moderate", description="风险偏好：conservative/moderate/aggressive")
    time_horizon: str = Field(default="medium", description="投资期限：short/medium/long")
    
    # 个性化参数
    personality_traits: Dict[str, Any] = Field(default_factory=dict, description="个性特征")
    favorite_indicators: List[str] = Field(default_factory=list, description="偏好指标")
    avoid_sectors: List[str] = Field(default_factory=list, description="回避行业")
    
    # 性能配置
    enable_caching: bool = Field(default=True, description="启用缓存")
    cache_ttl_seconds: int = Field(default=300, description="缓存有效期")
    max_retries: int = Field(default=3, description="最大重试次数")
    enable_fallback: bool = Field(default=True, description="启用降级策略")


class InvestmentMasterAgent(BaseAgent):
    """投资大师Agent基类"""
    
    def __init__(self, config: InvestmentMasterConfig, message_bus=None):
        """初始化投资大师Agent"""
        super().__init__(config, message_bus)
        
        self.master_config: InvestmentMasterConfig = config
        self.master_name = config.master_name
        self.investment_style = config.investment_style
        self.specialty = config.specialty
        
        # 初始化LLM客户端
        self.llm_client: Optional[LLMClient] = None
        
        # 初始化提示词模板
        self.prompt_template = PromptTemplate()
        self.style_prompts = InvestmentStylePrompts()
        
        # 分析缓存
        self._analysis_cache: Dict[str, Tuple[MasterInsight, float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # 性能监控
        self._llm_calls = 0
        self._llm_errors = 0
        self._total_tokens_used = 0
        self._avg_response_time = 0.0
        
        self.log_info(f"Initialized Investment Master Agent: {self.master_name}")
    
    async def _initialize(self) -> None:
        """初始化Agent"""
        # 创建LLM客户端
        llm_config = LLMConfig(
            provider=self.master_config.llm_provider,
            model=self.master_config.llm_model,
            temperature=self.master_config.llm_temperature,
            max_tokens=self.master_config.llm_max_tokens,
            timeout=self.master_config.llm_timeout
        )
        
        self.llm_client = LLMClient(llm_config)
        await self.llm_client.initialize()
        
        # 初始化个性化配置
        await self._setup_personality()
        
        self.log_info(f"{self.master_name} initialized with {self.investment_style.value} style")
    
    async def _setup_personality(self):
        """设置个性化配置"""
        # 根据投资大师设置默认个性
        if not self.master_config.personality_traits:
            self.master_config.personality_traits = self._get_default_personality()
        
        # 设置默认偏好指标
        if not self.master_config.favorite_indicators:
            self.master_config.favorite_indicators = self._get_default_indicators()
    
    def _get_default_personality(self) -> Dict[str, Any]:
        """获取默认个性特征"""
        personalities = {
            InvestmentStyle.VALUE: {
                "patience": "high",
                "risk_aversion": "moderate",
                "focus": "fundamentals",
                "decision_speed": "slow"
            },
            InvestmentStyle.GROWTH: {
                "patience": "moderate",
                "risk_aversion": "low",
                "focus": "future_potential",
                "decision_speed": "moderate"
            },
            InvestmentStyle.MOMENTUM: {
                "patience": "low",
                "risk_aversion": "low",
                "focus": "price_action",
                "decision_speed": "fast"
            },
            InvestmentStyle.CONTRARIAN: {
                "patience": "high",
                "risk_aversion": "moderate",
                "focus": "market_sentiment",
                "decision_speed": "moderate"
            }
        }
        return personalities.get(self.investment_style, {})
    
    def _get_default_indicators(self) -> List[str]:
        """获取默认偏好指标"""
        indicators = {
            InvestmentStyle.VALUE: ["PE", "PB", "ROE", "Dividend_Yield"],
            InvestmentStyle.GROWTH: ["Revenue_Growth", "EPS_Growth", "PEG"],
            InvestmentStyle.MOMENTUM: ["RSI", "MACD", "Volume", "Price_Change"],
            InvestmentStyle.TECHNICAL: ["MA", "Bollinger_Bands", "Support_Resistance"],
            InvestmentStyle.FUNDAMENTAL: ["FCF", "Debt_Ratio", "ROIC", "Margins"]
        }
        return indicators.get(self.investment_style, [])
    
    async def analyze(self, state: TradingState) -> List[Signal]:
        """分析市场并生成信号"""
        insights = await self.generate_insights(state)
        signals = self._convert_insights_to_signals(insights)
        return signals
    
    async def generate_insights(self, state: TradingState) -> List[MasterInsight]:
        """生成投资洞察"""
        insights = []
        
        # 对每个交易对进行分析
        for symbol in state.active_symbols:
            # 检查缓存
            if self.master_config.enable_caching:
                cached_insight = self._get_cached_insight(symbol)
                if cached_insight:
                    insights.append(cached_insight)
                    continue
            
            # 生成新的洞察
            try:
                insight = await self._analyze_symbol(symbol, state)
                insights.append(insight)
                
                # 更新缓存
                if self.master_config.enable_caching:
                    self._cache_insight(symbol, insight)
                
            except Exception as e:
                self.log_error(f"Failed to analyze {symbol}: {e}")
                
                # 使用降级策略
                if self.master_config.enable_fallback:
                    fallback_insight = self._generate_fallback_insight(symbol)
                    insights.append(fallback_insight)
        
        return insights
    
    async def _analyze_symbol(self, symbol: str, state: TradingState) -> MasterInsight:
        """分析单个交易对"""
        start_time = time.time()
        
        try:
            # 准备市场数据
            market_data = self._prepare_market_data(symbol, state)
            
            # 构建分析提示词
            prompt = self._build_analysis_prompt(symbol, market_data)
            
            # 调用LLM
            response = await self._call_llm(prompt)
            
            # 解析响应
            insight = self._parse_llm_response(response, symbol)
            
            # 更新性能指标
            elapsed = time.time() - start_time
            self._update_performance_metrics(elapsed, response.tokens_used)
            
            return insight
            
        except Exception as e:
            self._llm_errors += 1
            raise Exception(f"Analysis failed for {symbol}: {e}")
    
    def _prepare_market_data(self, symbol: str, state: TradingState) -> Dict[str, Any]:
        """准备市场数据"""
        market_data = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "price_data": {},
            "volume_data": {},
            "indicators": {},
            "news_sentiment": {},
            "market_conditions": {}
        }
        
        # 提取价格数据
        if symbol in state.market_data:
            data = state.market_data[symbol]
            market_data["price_data"] = {
                "current": data.close,
                "open": data.open,
                "high": data.high,
                "low": data.low,
                "change_pct": ((data.close - data.open) / data.open * 100) if data.open else 0
            }
            market_data["volume_data"] = {
                "current": data.volume,
                "average": data.volume  # TODO: 计算平均成交量
            }
        
        # 添加技术指标
        if hasattr(state, 'indicators') and symbol in state.indicators:
            market_data["indicators"] = state.indicators[symbol]
        
        # 添加市场情绪
        if hasattr(state, 'market_sentiment'):
            market_data["news_sentiment"] = state.market_sentiment
        
        return market_data
    
    def _build_analysis_prompt(self, symbol: str, market_data: Dict[str, Any]) -> str:
        """构建分析提示词"""
        # 获取风格特定的系统提示
        system_prompt = self.style_prompts.get_system_prompt(
            self.investment_style,
            self.master_name,
            self.master_config.personality_traits
        )
        
        # 构建分析请求
        analysis_request = self.prompt_template.build_analysis_prompt(
            symbol=symbol,
            market_data=market_data,
            analysis_type=AnalysisType.MARKET_TREND,
            time_horizon=self.master_config.time_horizon,
            risk_tolerance=self.master_config.risk_tolerance,
            favorite_indicators=self.master_config.favorite_indicators
        )
        
        return f"{system_prompt}\n\n{analysis_request}"
    
    async def _call_llm(self, prompt: str) -> LLMResponse:
        """调用LLM"""
        self._llm_calls += 1
        
        for attempt in range(self.master_config.max_retries):
            try:
                response = await self.llm_client.generate(
                    prompt=prompt,
                    system_prompt=f"You are {self.master_name}, a legendary investor with {self.investment_style.value} investment style."
                )
                return response
                
            except Exception as e:
                self.log_warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt == self.master_config.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # 指数退避
    
    def _parse_llm_response(self, response: LLMResponse, symbol: str) -> MasterInsight:
        """解析LLM响应"""
        try:
            # 尝试解析JSON格式
            if response.content.strip().startswith('{'):
                data = json.loads(response.content)
            else:
                # 使用自然语言解析
                data = self._parse_natural_language(response.content)
            
            return MasterInsight(
                master_name=self.master_name,
                investment_style=self.investment_style,
                analysis_type=AnalysisType(data.get("analysis_type", "market_trend")),
                main_conclusion=data.get("conclusion", ""),
                confidence_score=float(data.get("confidence", 0.5)),
                key_points=data.get("key_points", []),
                recommendations=data.get("recommendations", []),
                risk_warnings=data.get("risks", []),
                time_horizon=data.get("time_horizon", self.master_config.time_horizon),
                metadata={
                    "symbol": symbol,
                    "llm_model": response.model,
                    "tokens_used": response.tokens_used,
                    "response_time": response.response_time
                }
            )
            
        except Exception as e:
            self.log_error(f"Failed to parse LLM response: {e}")
            return self._generate_fallback_insight(symbol)
    
    def _parse_natural_language(self, content: str) -> Dict[str, Any]:
        """解析自然语言响应"""
        # 简单的关键词提取和情感分析
        result = {
            "analysis_type": "market_trend",
            "conclusion": "",
            "confidence": 0.5,
            "key_points": [],
            "recommendations": [],
            "risks": [],
            "time_horizon": "medium"
        }
        
        lines = content.strip().split('\n')
        
        for line in lines:
            line_lower = line.lower()
            
            # 提取结论
            if any(word in line_lower for word in ["conclusion", "summary", "overall"]):
                result["conclusion"] = line
            
            # 提取置信度
            if "confidence" in line_lower or "certain" in line_lower:
                # 尝试提取数字
                import re
                numbers = re.findall(r'\d+\.?\d*', line)
                if numbers:
                    conf = float(numbers[0])
                    if conf > 1:
                        conf = conf / 100
                    result["confidence"] = conf
            
            # 提取关键点
            if any(word in line_lower for word in ["key", "important", "note"]):
                result["key_points"].append(line)
            
            # 提取建议
            if any(word in line_lower for word in ["recommend", "suggest", "should"]):
                result["recommendations"].append({"action": line})
            
            # 提取风险
            if any(word in line_lower for word in ["risk", "warning", "caution"]):
                result["risks"].append(line)
        
        return result
    
    def _convert_insights_to_signals(self, insights: List[MasterInsight]) -> List[Signal]:
        """将洞察转换为交易信号"""
        signals = []
        
        for insight in insights:
            # 根据洞察生成信号
            for rec in insight.recommendations:
                signal = Signal(
                    source=f"{self.name}_{self.master_name}",
                    symbol=insight.metadata.get("symbol", ""),
                    action=self._map_recommendation_to_action(rec),
                    strength=self._calculate_signal_strength(insight),
                    confidence=insight.confidence_score,
                    reason=insight.main_conclusion,
                    metadata={
                        "master_name": self.master_name,
                        "investment_style": self.investment_style.value,
                        "analysis_type": insight.analysis_type.value,
                        "time_horizon": insight.time_horizon,
                        "key_points": insight.key_points,
                        "risk_warnings": insight.risk_warnings
                    }
                )
                
                if self.validate_signal(signal):
                    signals.append(signal)
        
        return signals
    
    def _map_recommendation_to_action(self, recommendation: Dict[str, Any]) -> str:
        """映射推荐到交易动作"""
        action_text = str(recommendation.get("action", "")).lower()
        
        if any(word in action_text for word in ["buy", "long", "accumulate"]):
            return "BUY"
        elif any(word in action_text for word in ["sell", "short", "reduce"]):
            return "SELL"
        elif any(word in action_text for word in ["hold", "wait", "maintain"]):
            return "HOLD"
        else:
            return "NEUTRAL"
    
    def _calculate_signal_strength(self, insight: MasterInsight) -> float:
        """计算信号强度"""
        base_strength = insight.confidence_score
        
        # 根据风险警告调整
        risk_penalty = len(insight.risk_warnings) * 0.1
        base_strength -= risk_penalty
        
        # 根据时间范围调整
        if insight.time_horizon == "short":
            base_strength *= 0.8
        elif insight.time_horizon == "long":
            base_strength *= 1.2
        
        # 限制在[-1, 1]范围内
        return max(-1.0, min(1.0, base_strength))
    
    def _get_cached_insight(self, symbol: str) -> Optional[MasterInsight]:
        """获取缓存的洞察"""
        if symbol in self._analysis_cache:
            insight, timestamp = self._analysis_cache[symbol]
            
            # 检查是否过期
            if time.time() - timestamp < self.master_config.cache_ttl_seconds:
                self._cache_hits += 1
                self.log_debug(f"Cache hit for {symbol}")
                return insight
        
        self._cache_misses += 1
        return None
    
    def _cache_insight(self, symbol: str, insight: MasterInsight):
        """缓存洞察"""
        self._analysis_cache[symbol] = (insight, time.time())
        
        # 限制缓存大小
        if len(self._analysis_cache) > 100:
            # 删除最旧的条目
            oldest_symbol = min(self._analysis_cache.keys(), 
                              key=lambda k: self._analysis_cache[k][1])
            del self._analysis_cache[oldest_symbol]
    
    def _generate_fallback_insight(self, symbol: str) -> MasterInsight:
        """生成降级洞察"""
        return MasterInsight(
            master_name=self.master_name,
            investment_style=self.investment_style,
            analysis_type=AnalysisType.MARKET_TREND,
            main_conclusion="Unable to perform detailed analysis, using conservative approach",
            confidence_score=0.3,
            key_points=["Analysis unavailable", "Using fallback strategy"],
            recommendations=[{"action": "HOLD", "reason": "Insufficient data"}],
            risk_warnings=["Analysis quality compromised"],
            time_horizon="short",
            metadata={"symbol": symbol, "fallback": True}
        )
    
    def _update_performance_metrics(self, response_time: float, tokens_used: int):
        """更新性能指标"""
        self._total_tokens_used += tokens_used
        
        # 更新平均响应时间
        n = self._llm_calls
        if n > 0:
            self._avg_response_time = (
                (self._avg_response_time * (n - 1) + response_time) / n
            )
    
    async def get_analysis_summary(self) -> Dict[str, Any]:
        """获取分析摘要"""
        return {
            "master_name": self.master_name,
            "investment_style": self.investment_style.value,
            "specialty": self.specialty,
            "performance": {
                "llm_calls": self._llm_calls,
                "llm_errors": self._llm_errors,
                "total_tokens": self._total_tokens_used,
                "avg_response_time": round(self._avg_response_time, 3),
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "cache_hit_rate": self._cache_hits / max(1, self._cache_hits + self._cache_misses)
            },
            "configuration": {
                "llm_provider": self.master_config.llm_provider,
                "llm_model": self.master_config.llm_model,
                "analysis_depth": self.master_config.analysis_depth,
                "risk_tolerance": self.master_config.risk_tolerance,
                "time_horizon": self.master_config.time_horizon
            }
        }
    
    async def update_master_config(self, **kwargs):
        """更新大师配置"""
        for key, value in kwargs.items():
            if hasattr(self.master_config, key):
                setattr(self.master_config, key, value)
                self.log_info(f"Updated master config: {key} = {value}")
        
        # 重新初始化个性化配置
        await self._setup_personality()
    
    @abstractmethod
    async def make_investment_decision(self, 
                                      state: TradingState,
                                      portfolio: Dict[str, Any]) -> FinalDecision:
        """做出投资决策（由子类实现）"""
        pass
    
    @abstractmethod
    async def evaluate_portfolio(self, 
                                portfolio: Dict[str, Any],
                                market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """评估投资组合（由子类实现）"""
        pass