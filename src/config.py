from pathlib import Path
from typing import Literal, Optional

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