from pathlib import Path
from typing import Literal, Optional, Dict, Any
import os

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Binance API
    binance_api_key: str = Field(default="", description="Binance API Key")
    binance_api_secret: str = Field(default="", description="Binance API Secret")
    binance_testnet: bool = Field(default=True, description="Use Binance Testnet")

    # Trading Configuration
    trading_mode: Literal["paper", "backtest", "live"] = Field(
        default="paper", description="Trading mode"
    )
    default_leverage: int = Field(default=1, ge=1, le=125)
    max_position_size: float = Field(default=1000.0, gt=0)

    # Risk Management
    max_daily_loss_percent: float = Field(default=5.0, gt=0, le=100)
    max_position_percent: float = Field(default=10.0, gt=0, le=100)
    stop_loss_percent: float = Field(default=2.0, gt=0, le=100)

    # System Configuration
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    data_dir: Path = Field(default=Path("./data"))
    log_dir: Path = Field(default=Path("./logs"))
    database_url: str = Field(default="sqlite:///./data/trading.db")
    duckdb_path: Path = Field(default=Path("./data/market_data.duckdb"))
    
    @property
    def data_directory(self) -> str:
        """获取数据目录路径"""
        return str(self.data_dir)

    # ZeroMQ Configuration
    zmq_pub_port: int = Field(default=5555, gt=1024, lt=65535)
    zmq_sub_port: int = Field(default=5556, gt=1024, lt=65535)

    # Web Interface
    web_host: str = Field(default="0.0.0.0")
    web_port: int = Field(default=8000, gt=1024, lt=65535)
    web_reload: bool = Field(default=True)

    # LLM Configuration
    openai_api_key: Optional[str] = Field(default=None)
    llm_model: str = Field(default="gpt-3.5-turbo")
    local_llm_enabled: bool = Field(default=False)
    local_llm_url: str = Field(default="http://localhost:11434")

    # Monitoring
    enable_metrics: bool = Field(default=True)
    metrics_port: int = Field(default=9090, gt=1024, lt=65535)
    
    # Environment
    environment: Literal["development", "testing", "production"] = Field(
        default="development", description="Current environment"
    )

    @field_validator("data_dir", "log_dir", "duckdb_path")
    @classmethod
    def create_directories(cls, v: Path) -> Path:
        if isinstance(v, str):
            v = Path(v)
        if v.suffix:  # If it's a file path
            v.parent.mkdir(parents=True, exist_ok=True)
        else:  # If it's a directory path
            v.mkdir(parents=True, exist_ok=True)
        return v

    @property
    def binance_base_url(self) -> str:
        if self.binance_testnet:
            return "https://testnet.binancefuture.com"
        return "https://fapi.binance.com"

    @property
    def binance_ws_url(self) -> str:
        if self.binance_testnet:
            return "wss://stream.binancefuture.com"
        return "wss://fstream.binance.com"


settings = Settings()


class Config:
    """简化的配置类，用于API和其他组件"""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        self._config = config_dict or {}
        self._load_env_vars()
    
    def _load_env_vars(self):
        """加载环境变量"""
        # 从环境变量加载常用配置
        env_mappings = {
            'api.host': 'API_HOST',
            'api.port': 'API_PORT', 
            'api.debug': 'API_DEBUG',
            'auth.jwt_secret_key': 'AUTH_JWT_SECRET_KEY',
            'auth.access_token_expire_minutes': 'AUTH_ACCESS_TOKEN_EXPIRE_MINUTES',
            'database.url': 'DATABASE_URL',
        }
        
        for config_key, env_key in env_mappings.items():
            if env_key in os.environ:
                value = os.environ[env_key]
                # 类型转换
                if env_key in ['API_PORT', 'AUTH_ACCESS_TOKEN_EXPIRE_MINUTES']:
                    try:
                        value = int(value)
                    except ValueError:
                        pass
                elif env_key == 'API_DEBUG':
                    value = value.lower() in ('true', '1', 'yes', 'on')
                
                self.set(config_key, value)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值，支持点号分隔的嵌套键"""
        try:
            value = self._config
            for part in key.split('.'):
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """设置配置值，支持点号分隔的嵌套键"""
        parts = key.split('.')
        config = self._config
        
        for part in parts[:-1]:
            if part not in config:
                config[part] = {}
            config = config[part]
        
        config[parts[-1]] = value
    
    def update(self, other_config: Dict[str, Any]):
        """更新配置"""
        self._config.update(other_config)
    
    @classmethod
    def load_from_env(cls) -> 'Config':
        """从环境变量加载配置"""
        return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self._config.copy()