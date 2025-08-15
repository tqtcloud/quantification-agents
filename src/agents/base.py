from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.core.models import MarketData, Order, Position, Signal, TradingState
from src.utils.logger import LoggerMixin


class AgentConfig(BaseModel):
    """Base configuration for all agents."""
    name: str
    enabled: bool = True
    priority: int = Field(default=0, ge=0, le=100)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class BaseAgent(ABC, LoggerMixin):
    """Abstract base class for all trading agents."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.name = config.name
        self.enabled = config.enabled
        self.priority = config.priority
        self._initialized = False
        
    async def initialize(self) -> None:
        """Initialize the agent."""
        if self._initialized:
            return
            
        self.log_info(f"Initializing agent: {self.name}")
        await self._initialize()
        self._initialized = True
        self.log_info(f"Agent initialized: {self.name}")
    
    @abstractmethod
    async def _initialize(self) -> None:
        """Agent-specific initialization logic."""
        pass
    
    async def shutdown(self) -> None:
        """Shutdown the agent."""
        if not self._initialized:
            return
            
        self.log_info(f"Shutting down agent: {self.name}")
        await self._shutdown()
        self._initialized = False
        self.log_info(f"Agent shutdown: {self.name}")
    
    async def _shutdown(self) -> None:
        """Agent-specific shutdown logic."""
        pass
    
    @abstractmethod
    async def analyze(self, state: TradingState) -> List[Signal]:
        """
        Analyze the current trading state and generate signals.
        
        Args:
            state: Current trading state
            
        Returns:
            List of trading signals
        """
        pass
    
    async def validate_signal(self, signal: Signal) -> bool:
        """
        Validate a trading signal before execution.
        
        Args:
            signal: Signal to validate
            
        Returns:
            True if signal is valid, False otherwise
        """
        return (
            signal.strength != 0 and
            abs(signal.strength) >= self.config.parameters.get("min_signal_strength", 0.3) and
            signal.confidence >= self.config.parameters.get("min_confidence", 0.5)
        )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, enabled={self.enabled})"


class AnalysisAgent(BaseAgent):
    """Base class for analysis agents that generate signals."""
    
    async def _initialize(self) -> None:
        """Initialize analysis agent."""
        pass
    
    @abstractmethod
    async def analyze(self, state: TradingState) -> List[Signal]:
        """Analyze market and generate signals."""
        pass


class ExecutionAgent(BaseAgent):
    """Base class for execution agents that manage orders."""
    
    async def _initialize(self) -> None:
        """Initialize execution agent."""
        pass
    
    @abstractmethod
    async def execute(self, signal: Signal, state: TradingState) -> Optional[Order]:
        """
        Execute a trading signal.
        
        Args:
            signal: Signal to execute
            state: Current trading state
            
        Returns:
            Order if execution successful, None otherwise
        """
        pass
    
    async def analyze(self, state: TradingState) -> List[Signal]:
        """Execution agents don't generate signals."""
        return []


class RiskAgent(BaseAgent):
    """Base class for risk management agents."""
    
    async def _initialize(self) -> None:
        """Initialize risk agent."""
        pass
    
    @abstractmethod
    async def check_risk(self, order: Order, state: TradingState) -> bool:
        """
        Check if an order passes risk management rules.
        
        Args:
            order: Order to check
            state: Current trading state
            
        Returns:
            True if order is allowed, False otherwise
        """
        pass
    
    @abstractmethod
    async def adjust_position_size(self, order: Order, state: TradingState) -> Order:
        """
        Adjust order size based on risk management rules.
        
        Args:
            order: Order to adjust
            state: Current trading state
            
        Returns:
            Adjusted order
        """
        pass
    
    async def analyze(self, state: TradingState) -> List[Signal]:
        """Risk agents may generate risk-related signals."""
        signals = []
        
        # Check for high-risk conditions
        if state.risk_metrics:
            if state.risk_metrics.margin_usage > 0.8:
                signals.append(Signal(
                    source=self.name,
                    symbol="ALL",
                    action="REDUCE_EXPOSURE",
                    strength=-0.8,
                    confidence=0.9,
                    reason="High margin usage detected",
                    metadata={"margin_usage": state.risk_metrics.margin_usage}
                ))
            
            if state.risk_metrics.current_drawdown > 0.1:
                signals.append(Signal(
                    source=self.name,
                    symbol="ALL",
                    action="STOP_TRADING",
                    strength=-1.0,
                    confidence=0.95,
                    reason="Maximum drawdown exceeded",
                    metadata={"drawdown": state.risk_metrics.current_drawdown}
                ))
        
        return signals