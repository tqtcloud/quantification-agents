from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

from src.config import settings
from src.core.models import OrderSide, OrderStatus, OrderType, PositionSide

Base = declarative_base()


class TradingSession(Base):
    __tablename__ = "trading_sessions"
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(50), unique=True, nullable=False)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    mode = Column(String(20), nullable=False)  # paper, backtest, live
    initial_balance = Column(Float, nullable=False)
    final_balance = Column(Float, nullable=True)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    max_drawdown = Column(Float, default=0.0)
    sharpe_ratio = Column(Float, nullable=True)
    metadata = Column(JSON, default={})
    
    orders = relationship("OrderRecord", back_populates="session")
    positions = relationship("PositionRecord", back_populates="session")
    trades = relationship("TradeRecord", back_populates="session")


class OrderRecord(Base):
    __tablename__ = "orders"
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("trading_sessions.id"))
    order_id = Column(String(50), unique=True, nullable=False)
    client_order_id = Column(String(50), nullable=True)
    symbol = Column(String(20), nullable=False)
    side = Column(Enum(OrderSide), nullable=False)
    order_type = Column(Enum(OrderType), nullable=False)
    quantity = Column(Float, nullable=False)
    price = Column(Float, nullable=True)
    stop_price = Column(Float, nullable=True)
    status = Column(Enum(OrderStatus), nullable=False)
    executed_qty = Column(Float, default=0.0)
    avg_price = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    metadata = Column(JSON, default={})
    
    session = relationship("TradingSession", back_populates="orders")


class PositionRecord(Base):
    __tablename__ = "positions"
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("trading_sessions.id"))
    symbol = Column(String(20), nullable=False)
    side = Column(Enum(PositionSide), nullable=False)
    quantity = Column(Float, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    realized_pnl = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    max_profit = Column(Float, default=0.0)
    max_loss = Column(Float, default=0.0)
    opened_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime, nullable=True)
    metadata = Column(JSON, default={})
    
    session = relationship("TradingSession", back_populates="positions")


class TradeRecord(Base):
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("trading_sessions.id"))
    trade_id = Column(String(50), unique=True, nullable=False)
    order_id = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False)
    side = Column(Enum(OrderSide), nullable=False)
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    commission = Column(Float, default=0.0)
    commission_asset = Column(String(10), nullable=True)
    realized_pnl = Column(Float, default=0.0)
    executed_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON, default={})
    
    session = relationship("TradingSession", back_populates="trades")


class SignalRecord(Base):
    __tablename__ = "signals"
    
    id = Column(Integer, primary_key=True)
    source = Column(String(50), nullable=False)
    symbol = Column(String(20), nullable=False)
    action = Column(String(20), nullable=False)
    strength = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    reason = Column(Text, nullable=True)
    executed = Column(Boolean, default=False)
    order_id = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    metadata = Column(JSON, default={})


class StrategyRecord(Base):
    __tablename__ = "strategies"
    
    id = Column(Integer, primary_key=True)
    strategy_id = Column(String(50), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    version = Column(String(20), nullable=False)
    description = Column(Text, nullable=True)
    parameters = Column(JSON, default={})
    performance_metrics = Column(JSON, default={})
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)


class BacktestResult(Base):
    __tablename__ = "backtest_results"
    
    id = Column(Integer, primary_key=True)
    backtest_id = Column(String(50), unique=True, nullable=False)
    strategy_id = Column(String(50), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Float, nullable=False)
    final_capital = Column(Float, nullable=False)
    total_return = Column(Float, nullable=False)
    annual_return = Column(Float, nullable=False)
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=False)
    win_rate = Column(Float, nullable=False)
    profit_factor = Column(Float, nullable=True)
    total_trades = Column(Integer, nullable=False)
    parameters = Column(JSON, default={})
    metrics = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)


# Database initialization
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
    pool_pre_ping=True,
    echo=settings.log_level == "DEBUG",
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_database():
    """Initialize the database tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()