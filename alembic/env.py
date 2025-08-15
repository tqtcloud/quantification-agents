from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.database import Base, TradingEnvironment
from src.config import settings

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

def get_url_for_environment(environment: str):
    """获取指定环境的数据库URL"""
    data_dir = settings.data_directory
    if environment == 'testnet':
        return f"sqlite:///{data_dir}/testnet/trading_testnet.db"
    elif environment == 'mainnet':
        return f"sqlite:///{data_dir}/mainnet/trading_mainnet.db"
    elif environment == 'paper':
        return f"sqlite:///{data_dir}/paper/trading_paper.db"
    else:
        return f"sqlite:///{data_dir}/testnet/trading_testnet.db"

def run_migrations_offline():
    """Run migrations in 'offline' mode.
    
    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.
    
    Calls to context.execute() here emit the given string to the
    script output.
    """
    environment = os.environ.get('MIGRATION_ENV', 'testnet')
    url = get_url_for_environment(environment)
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode.
    
    In this scenario we need to create an Engine
    and associate a connection with the context.
    """
    environment = os.environ.get('MIGRATION_ENV', 'testnet')
    url = get_url_for_environment(environment)
    
    configuration = config.get_section(config.config_ini_section)
    configuration['sqlalchemy.url'] = url
    
    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, 
            target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()