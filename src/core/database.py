import asyncio
import os
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime
from decimal import Decimal
from enum import Enum as PyEnum
from typing import Optional, Dict, Any, Generator, AsyncGenerator
from pathlib import Path

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
    MetaData,
    event,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, Session
from sqlalchemy.pool import StaticPool, QueuePool
from alembic.config import Config
from alembic import command
from alembic.script import ScriptDirectory
from alembic.runtime.migration import MigrationContext

from src.config import settings
from src.core.models import OrderSide, OrderStatus, OrderType, PositionSide
from src.exchanges.trading_interface import TradingEnvironment
from src.utils.logger import LoggerMixin

Base = declarative_base()


class EnvironmentMetadata(Base):
    """环境元数据表 - 追踪不同交易环境的配置"""
    __tablename__ = "environment_metadata"
    
    id = Column(Integer, primary_key=True)
    environment = Column(Enum(TradingEnvironment), nullable=False, unique=True)
    database_path = Column(String(255), nullable=True)
    api_endpoint = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    metadata = Column(JSON, default={})


class TradingSession(Base):
    __tablename__ = "trading_sessions"
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(50), unique=True, nullable=False)
    environment = Column(Enum(TradingEnvironment), nullable=False, default=TradingEnvironment.TESTNET)
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
    environment = Column(Enum(TradingEnvironment), nullable=False, default=TradingEnvironment.TESTNET)
    order_id = Column(String(50), nullable=False)
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
    environment = Column(Enum(TradingEnvironment), nullable=False, default=TradingEnvironment.TESTNET)
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
    environment = Column(Enum(TradingEnvironment), nullable=False, default=TradingEnvironment.TESTNET)
    trade_id = Column(String(50), nullable=False)
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
    environment = Column(Enum(TradingEnvironment), nullable=False, default=TradingEnvironment.TESTNET)
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


class DatabaseManager(LoggerMixin):
    """数据库管理器 - 支持多环境数据隔离"""
    
    def __init__(self):
        self.engines: Dict[TradingEnvironment, Any] = {}
        self.session_makers: Dict[TradingEnvironment, Any] = {}
        self._current_environment: Optional[TradingEnvironment] = None
        self._data_dir = Path(settings.data_directory)
        self._init_directories()
    
    def _init_directories(self):
        """初始化数据目录"""
        self._data_dir.mkdir(parents=True, exist_ok=True)
        for env in TradingEnvironment:
            env_dir = self._data_dir / env.value
            env_dir.mkdir(exist_ok=True)
    
    def _get_database_url(self, environment: TradingEnvironment) -> str:
        """获取环境特定的数据库URL"""
        if environment == TradingEnvironment.TESTNET:
            db_path = self._data_dir / "testnet" / "trading_testnet.db"
        elif environment == TradingEnvironment.MAINNET:
            db_path = self._data_dir / "mainnet" / "trading_mainnet.db"
        else:  # PAPER
            db_path = self._data_dir / "paper" / "trading_paper.db"
        
        return f"sqlite:///{db_path}"
    
    def get_engine(self, environment: TradingEnvironment):
        """获取环境特定的数据库引擎"""
        if environment not in self.engines:
            database_url = self._get_database_url(environment)
            
            self.engines[environment] = create_engine(
                database_url,
                connect_args={
                    "check_same_thread": False,
                    "timeout": 30,
                },
                poolclass=StaticPool,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=settings.log_level == "DEBUG",
            )
            
            # 配置SQLite WAL模式以提高并发性能
            @event.listens_for(self.engines[environment], "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=10000")
                cursor.execute("PRAGMA temp_store=MEMORY")
                cursor.close()
        
        return self.engines[environment]
    
    def get_session_maker(self, environment: TradingEnvironment):
        """获取环境特定的会话创建器"""
        if environment not in self.session_makers:
            engine = self.get_engine(environment)
            self.session_makers[environment] = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=engine
            )
        
        return self.session_makers[environment]
    
    def init_database(self, environment: TradingEnvironment):
        """初始化指定环境的数据库表"""
        engine = self.get_engine(environment)
        Base.metadata.create_all(bind=engine)
        
        # 初始化环境元数据
        session_maker = self.get_session_maker(environment)
        with session_maker() as session:
            env_metadata = session.query(EnvironmentMetadata).filter_by(
                environment=environment
            ).first()
            
            if not env_metadata:
                env_metadata = EnvironmentMetadata(
                    environment=environment,
                    database_path=str(self._get_database_url(environment)),
                    is_active=True
                )
                session.add(env_metadata)
                session.commit()
        
        self.log_info(f"Database initialized for environment: {environment.value}")
    
    def init_all_databases(self):
        """初始化所有环境的数据库"""
        for environment in TradingEnvironment:
            self.init_database(environment)
    
    @contextmanager
    def get_session(self, environment: TradingEnvironment) -> Generator[Session, None, None]:
        """获取数据库会话的上下文管理器"""
        session_maker = self.get_session_maker(environment)
        session = session_maker()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.log_error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def set_current_environment(self, environment: TradingEnvironment):
        """设置当前活跃环境"""
        self._current_environment = environment
        self.log_debug(f"Current database environment set to: {environment.value}")
    
    def get_current_environment(self) -> Optional[TradingEnvironment]:
        """获取当前活跃环境"""
        return self._current_environment
    
    @contextmanager
    def use_environment(self, environment: TradingEnvironment):
        """临时切换环境的上下文管理器"""
        previous_env = self._current_environment
        self.set_current_environment(environment)
        try:
            yield
        finally:
            self._current_environment = previous_env
    
    def close_all(self):
        """关闭所有数据库连接"""
        for environment, engine in self.engines.items():
            engine.dispose()
            self.log_info(f"Closed database connection for {environment.value}")
        
        self.engines.clear()
        self.session_makers.clear()
    
    def run_migrations(self, environment: TradingEnvironment, message: str = None):
        """运行数据库迁移"""
        try:
            # 设置环境变量
            os.environ['MIGRATION_ENV'] = environment.value
            
            alembic_cfg = Config(Path(__file__).parent.parent.parent / "alembic.ini")
            
            # 创建初始迁移（如果不存在）
            script_dir = ScriptDirectory.from_config(alembic_cfg)
            
            # 检查是否需要创建初始迁移
            engine = self.get_engine(environment)
            with engine.connect() as connection:
                migration_context = MigrationContext.configure(connection)
                current_rev = migration_context.get_current_revision()
                
                if current_rev is None:
                    # 创建初始迁移
                    command.revision(alembic_cfg, autogenerate=True, 
                                   message=message or f"Initial migration for {environment.value}")
                    self.log_info(f"Created initial migration for {environment.value}")
                
                # 运行迁移
                command.upgrade(alembic_cfg, "head")
                self.log_info(f"Applied migrations for {environment.value}")
                
        except Exception as e:
            self.log_error(f"Migration failed for {environment.value}: {e}")
            raise
        finally:
            # 清理环境变量
            os.environ.pop('MIGRATION_ENV', None)
    
    def create_migration(self, message: str, environment: TradingEnvironment = TradingEnvironment.TESTNET):
        """创建新的迁移文件"""
        try:
            os.environ['MIGRATION_ENV'] = environment.value
            
            alembic_cfg = Config(Path(__file__).parent.parent.parent / "alembic.ini")
            command.revision(alembic_cfg, autogenerate=True, message=message)
            
            self.log_info(f"Created migration: {message}")
            
        except Exception as e:
            self.log_error(f"Failed to create migration: {e}")
            raise
        finally:
            os.environ.pop('MIGRATION_ENV', None)
    
    def get_migration_history(self, environment: TradingEnvironment):
        """获取迁移历史"""
        try:
            os.environ['MIGRATION_ENV'] = environment.value
            
            alembic_cfg = Config(Path(__file__).parent.parent.parent / "alembic.ini")
            script_dir = ScriptDirectory.from_config(alembic_cfg)
            
            engine = self.get_engine(environment)
            with engine.connect() as connection:
                migration_context = MigrationContext.configure(connection)
                current_rev = migration_context.get_current_revision()
                
                revisions = []
                for revision in script_dir.walk_revisions():
                    revisions.append({
                        'revision': revision.revision,
                        'down_revision': revision.down_revision,
                        'message': revision.doc,
                        'is_current': revision.revision == current_rev
                    })
                
                return revisions
                
        except Exception as e:
            self.log_error(f"Failed to get migration history: {e}")
            return []
        finally:
            os.environ.pop('MIGRATION_ENV', None)


class DatabaseRepository(LoggerMixin):
    """数据访问层基类"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def _get_environment(self) -> TradingEnvironment:
        """获取当前环境，如果未设置则使用默认值"""
        env = self.db_manager.get_current_environment()
        if env is None:
            env = TradingEnvironment.TESTNET
            self.log_warning(f"No current environment set, using default: {env.value}")
        return env
    
    @contextmanager
    def session(self, environment: Optional[TradingEnvironment] = None) -> Generator[Session, None, None]:
        """获取数据库会话"""
        env = environment or self._get_environment()
        with self.db_manager.get_session(env) as session:
            yield session


class TradingSessionRepository(DatabaseRepository):
    """交易会话数据访问层"""
    
    def create_session(self, session_data: Dict[str, Any], environment: Optional[TradingEnvironment] = None) -> TradingSession:
        """创建交易会话"""
        env = environment or self._get_environment()
        
        with self.session(env) as db_session:
            trading_session = TradingSession(
                environment=env,
                **session_data
            )
            db_session.add(trading_session)
            db_session.flush()
            db_session.refresh(trading_session)
            return trading_session
    
    def get_sessions_by_environment(self, environment: Optional[TradingEnvironment] = None):
        """获取指定环境的所有会话"""
        env = environment or self._get_environment()
        
        with self.session(env) as db_session:
            return db_session.query(TradingSession).filter_by(environment=env).all()
    
    def get_active_session(self, environment: Optional[TradingEnvironment] = None) -> Optional[TradingSession]:
        """获取活跃的交易会话"""
        env = environment or self._get_environment()
        
        with self.session(env) as db_session:
            return db_session.query(TradingSession).filter_by(
                environment=env,
                end_time=None
            ).first()


class OrderRepository(DatabaseRepository):
    """订单数据访问层"""
    
    def create_order(self, order_data: Dict[str, Any], environment: Optional[TradingEnvironment] = None) -> OrderRecord:
        """创建订单记录"""
        env = environment or self._get_environment()
        
        with self.session(env) as db_session:
            order = OrderRecord(
                environment=env,
                **order_data
            )
            db_session.add(order)
            db_session.flush()
            db_session.refresh(order)
            return order
    
    def get_orders_by_environment(self, environment: Optional[TradingEnvironment] = None):
        """获取指定环境的所有订单"""
        env = environment or self._get_environment()
        
        with self.session(env) as db_session:
            return db_session.query(OrderRecord).filter_by(environment=env).all()
    
    def update_order_status(self, order_id: str, status: OrderStatus, 
                           environment: Optional[TradingEnvironment] = None):
        """更新订单状态"""
        env = environment or self._get_environment()
        
        with self.session(env) as db_session:
            order = db_session.query(OrderRecord).filter_by(
                environment=env,
                order_id=order_id
            ).first()
            
            if order:
                order.status = status
                order.updated_at = datetime.utcnow()
                db_session.flush()
                return order
            return None


# 全局数据库管理器实例
db_manager = DatabaseManager()

# 数据访问层实例
session_repo = TradingSessionRepository(db_manager)
order_repo = OrderRepository(db_manager)


# 兼容性函数
def init_database():
    """初始化数据库 - 兼容性函数"""
    db_manager.init_all_databases()


def get_db(environment: TradingEnvironment = TradingEnvironment.TESTNET):
    """获取数据库会话 - 兼容性函数"""
    with db_manager.get_session(environment) as session:
        yield session