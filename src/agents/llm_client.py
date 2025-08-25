"""
LLM客户端封装
支持多种大模型提供商：OpenAI、火山方舟、文心一言、通义千问等
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.logger import LoggerMixin


class LLMProvider(Enum):
    """LLM提供商枚举"""
    OPENAI = "openai"
    VOLCANO = "volcano"  # 火山方舟
    BAIDU = "baidu"  # 文心一言
    ALIBABA = "alibaba"  # 通义千问
    ANTHROPIC = "anthropic"  # Claude
    LOCAL = "local"  # 本地模型


@dataclass
class LLMResponse:
    """LLM响应"""
    content: str
    model: str
    provider: str
    tokens_used: int
    response_time: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LLMConfig(BaseModel):
    """LLM配置"""
    provider: str = Field(default="openai")
    model: str = Field(default="gpt-4")
    api_key: Optional[str] = Field(default=None)
    api_base: Optional[str] = Field(default=None)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2000, ge=100, le=8000)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    timeout: float = Field(default=30.0)
    max_retries: int = Field(default=3)
    
    # 提供商特定配置
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class BaseLLMProvider(ABC, LoggerMixin):
    """LLM提供商基类"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def initialize(self):
        """初始化提供商"""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def shutdown(self):
        """关闭提供商"""
        if self.session:
            await self.session.close()
    
    @abstractmethod
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """生成响应"""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """计算token数量"""
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI提供商"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_base = config.api_base or "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """调用OpenAI API"""
        start_time = time.time()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty
        }
        
        try:
            async with self.session.post(
                f"{self.api_base}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                data = await response.json()
                
                if response.status != 200:
                    error_msg = data.get("error", {}).get("message", "Unknown error")
                    raise Exception(f"OpenAI API error: {error_msg}")
                
                content = data["choices"][0]["message"]["content"]
                tokens_used = data["usage"]["total_tokens"]
                
                return LLMResponse(
                    content=content,
                    model=self.config.model,
                    provider="openai",
                    tokens_used=tokens_used,
                    response_time=time.time() - start_time,
                    metadata={"usage": data["usage"]}
                )
                
        except asyncio.TimeoutError:
            raise Exception(f"OpenAI API timeout after {self.config.timeout}s")
        except Exception as e:
            self.log_error(f"OpenAI API error: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """估算token数量"""
        # 简单估算：平均每个token约4个字符
        return len(text) // 4


class VolcanoProvider(BaseLLMProvider):
    """火山方舟提供商"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_base = config.api_base or "https://ark.cn-beijing.volcanicengine.com/api/v3"
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """调用火山方舟API"""
        start_time = time.time()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p
        }
        
        # 添加额外参数
        if self.config.extra_params:
            payload.update(self.config.extra_params)
        
        try:
            async with self.session.post(
                f"{self.api_base}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                data = await response.json()
                
                if response.status != 200:
                    error_msg = data.get("error", {}).get("message", "Unknown error")
                    raise Exception(f"Volcano API error: {error_msg}")
                
                content = data["choices"][0]["message"]["content"]
                tokens_used = data.get("usage", {}).get("total_tokens", 0)
                
                return LLMResponse(
                    content=content,
                    model=self.config.model,
                    provider="volcano",
                    tokens_used=tokens_used,
                    response_time=time.time() - start_time,
                    metadata={"usage": data.get("usage", {})}
                )
                
        except asyncio.TimeoutError:
            raise Exception(f"Volcano API timeout after {self.config.timeout}s")
        except Exception as e:
            self.log_error(f"Volcano API error: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """估算token数量"""
        # 中文字符平均1.5个token，英文4个字符1个token
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        english_chars = len(text) - chinese_chars
        return int(chinese_chars * 1.5 + english_chars / 4)


class BaiduProvider(BaseLLMProvider):
    """百度文心一言提供商"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.api_base = config.api_base or "https://aip.baidubce.com"
        self.access_token = None
        self.token_expire_time = 0
    
    async def _get_access_token(self):
        """获取访问令牌"""
        if self.access_token and time.time() < self.token_expire_time:
            return self.access_token
        
        # 获取新的access_token
        api_key = self.config.api_key
        secret_key = self.config.extra_params.get("secret_key", "")
        
        url = f"https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": api_key,
            "client_secret": secret_key
        }
        
        async with self.session.post(url, params=params) as response:
            data = await response.json()
            self.access_token = data["access_token"]
            self.token_expire_time = time.time() + data["expires_in"] - 60
            return self.access_token
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """调用文心一言API"""
        start_time = time.time()
        
        access_token = await self._get_access_token()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # 根据模型选择API端点
        model_endpoints = {
            "ernie-bot": "/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions",
            "ernie-bot-turbo": "/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/eb-instant",
            "ernie-bot-4": "/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro"
        }
        
        endpoint = model_endpoints.get(self.config.model, model_endpoints["ernie-bot"])
        url = f"{self.api_base}{endpoint}?access_token={access_token}"
        
        payload = {
            "messages": messages,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p
        }
        
        try:
            async with self.session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            ) as response:
                data = await response.json()
                
                if "error_code" in data:
                    raise Exception(f"Baidu API error: {data.get('error_msg', 'Unknown error')}")
                
                content = data["result"]
                tokens_used = data.get("usage", {}).get("total_tokens", 0)
                
                return LLMResponse(
                    content=content,
                    model=self.config.model,
                    provider="baidu",
                    tokens_used=tokens_used,
                    response_time=time.time() - start_time,
                    metadata={"usage": data.get("usage", {})}
                )
                
        except asyncio.TimeoutError:
            raise Exception(f"Baidu API timeout after {self.config.timeout}s")
        except Exception as e:
            self.log_error(f"Baidu API error: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """估算token数量"""
        return int(len(text) * 1.3)


class LocalProvider(BaseLLMProvider):
    """本地模型提供商（用于测试和开发）"""
    
    async def generate(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """生成模拟响应"""
        start_time = time.time()
        
        # 模拟延迟
        await asyncio.sleep(0.5)
        
        # 生成模拟响应
        content = self._generate_mock_response(prompt, system_prompt)
        
        return LLMResponse(
            content=content,
            model="local-mock",
            provider="local",
            tokens_used=self.count_tokens(prompt) + self.count_tokens(content),
            response_time=time.time() - start_time,
            metadata={"mock": True}
        )
    
    def _generate_mock_response(self, prompt: str, system_prompt: Optional[str]) -> str:
        """生成模拟响应内容"""
        # 简单的模板响应
        response = {
            "analysis_type": "market_trend",
            "conclusion": "Based on the current market conditions, the trend appears neutral with slight bullish bias.",
            "confidence": 0.65,
            "key_points": [
                "Market volume is increasing",
                "Technical indicators show mixed signals",
                "Support level holding strong"
            ],
            "recommendations": [
                {"action": "HOLD", "reason": "Wait for clearer signals"},
                {"action": "Consider small BUY", "reason": "If price breaks resistance"}
            ],
            "risks": [
                "Market volatility remains high",
                "Potential regulatory changes"
            ],
            "time_horizon": "medium"
        }
        
        return json.dumps(response, indent=2)
    
    def count_tokens(self, text: str) -> int:
        """估算token数量"""
        return len(text) // 4


class LLMClient(LoggerMixin):
    """统一的LLM客户端"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider: Optional[BaseLLMProvider] = None
        
        # 统计信息
        self.total_calls = 0
        self.total_tokens = 0
        self.total_cost = 0.0
    
    async def initialize(self):
        """初始化客户端"""
        # 根据配置创建提供商
        provider_map = {
            "openai": OpenAIProvider,
            "volcano": VolcanoProvider,
            "baidu": BaiduProvider,
            "local": LocalProvider
        }
        
        provider_class = provider_map.get(self.config.provider)
        if not provider_class:
            raise ValueError(f"Unsupported provider: {self.config.provider}")
        
        self.provider = provider_class(self.config)
        await self.provider.initialize()
        
        self.log_info(f"Initialized LLM client with {self.config.provider} provider")
    
    async def shutdown(self):
        """关闭客户端"""
        if self.provider:
            await self.provider.shutdown()
    
    async def generate(self, 
                      prompt: str, 
                      system_prompt: Optional[str] = None,
                      **kwargs) -> LLMResponse:
        """生成响应"""
        if not self.provider:
            raise RuntimeError("LLM client not initialized")
        
        try:
            # 合并额外参数到配置
            if kwargs:
                original_params = self.config.extra_params.copy()
                self.config.extra_params.update(kwargs)
            
            # 调用提供商
            response = await self.provider.generate(prompt, system_prompt)
            
            # 更新统计
            self.total_calls += 1
            self.total_tokens += response.tokens_used
            self.total_cost += self._calculate_cost(response.tokens_used)
            
            # 恢复原始参数
            if kwargs:
                self.config.extra_params = original_params
            
            return response
            
        except Exception as e:
            self.log_error(f"Failed to generate response: {e}")
            raise
    
    def count_tokens(self, text: str) -> int:
        """计算token数量"""
        if not self.provider:
            raise RuntimeError("LLM client not initialized")
        return self.provider.count_tokens(text)
    
    def _calculate_cost(self, tokens: int) -> float:
        """计算成本（美元）"""
        # 简单的成本估算
        cost_per_1k_tokens = {
            "gpt-4": 0.03,
            "gpt-3.5-turbo": 0.002,
            "claude-3": 0.025,
            "ernie-bot": 0.012,
            "local": 0.0
        }
        
        model_cost = cost_per_1k_tokens.get(self.config.model, 0.01)
        return (tokens / 1000) * model_cost
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "provider": self.config.provider,
            "model": self.config.model,
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 4),
            "avg_tokens_per_call": self.total_tokens // max(1, self.total_calls)
        }
    
    async def test_connection(self) -> bool:
        """测试连接"""
        try:
            response = await self.generate(
                "Hello, please respond with 'OK' if you can receive this message.",
                system_prompt="You are a helpful assistant for testing connections."
            )
            return "ok" in response.content.lower()
        except Exception as e:
            self.log_error(f"Connection test failed: {e}")
            return False