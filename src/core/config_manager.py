"""配置管理器 - 支持多环境配置和热更新"""

import asyncio
import json
import os
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from datetime import datetime
import yaml
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from ..config import settings
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ConfigChange:
    """配置变更记录"""
    timestamp: datetime
    key: str
    old_value: Any
    new_value: Any
    source: str  # file, api, environment


class ConfigFileHandler(FileSystemEventHandler):
    """配置文件变更处理器"""
    
    def __init__(self, config_manager: 'ConfigManager'):
        self.config_manager = config_manager
    
    def on_modified(self, event):
        if not event.is_directory:
            self.config_manager._handle_file_change(event.src_path)


class ConfigManager:
    """配置管理器 - 支持多环境配置和热更新"""
    
    def __init__(self, base_config_dir: str = "config"):
        self.base_config_dir = Path(base_config_dir)
        self.base_config_dir.mkdir(exist_ok=True)
        
        # 配置存储
        self.configs: Dict[str, Any] = {}
        self.environment_configs: Dict[str, Dict[str, Any]] = {}
        self.change_history: List[ConfigChange] = []
        
        # 监听器和回调
        self.file_observer: Optional[Observer] = None
        self.change_callbacks: List[Callable] = []
        self.lock = threading.RLock()
        
        # 当前环境
        self.current_environment = os.getenv('ENVIRONMENT', 'development')
        
        # 初始化配置文件
        self._init_config_files()
        self._load_all_configs()
    
    def _init_config_files(self):
        """初始化配置文件"""
        environments = ['development', 'testing', 'production']
        
        for env in environments:
            config_file = self.base_config_dir / f"{env}.yaml"
            if not config_file.exists():
                default_config = self._get_default_config(env)
                with open(config_file, 'w', encoding='utf-8') as f:
                    yaml.dump(default_config, f, default_flow_style=False, allow_unicode=True)
                logger.info(f"创建默认配置文件: {config_file}")
    
    def _get_default_config(self, environment: str) -> Dict[str, Any]:
        """获取默认配置"""
        base_config = {
            'environment': environment,
            'trading': {
                'mode': 'paper' if environment != 'production' else 'live',
                'max_position_size': 1000.0,
                'max_daily_loss_percent': 5.0,
                'max_position_percent': 10.0,
                'stop_loss_percent': 2.0,
                'leverage': 1
            },
            'binance': {
                'testnet': environment != 'production',
                'rate_limit': {
                    'requests_per_minute': 1200,
                    'orders_per_second': 10
                }
            },
            'database': {
                'url': f"sqlite:///./data/{environment}/trading.db",
                'duckdb_path': f"./data/{environment}/market_data.duckdb",
                'pool_size': 10,
                'backup_enabled': True
            },
            'logging': {
                'level': 'DEBUG' if environment == 'development' else 'INFO',
                'file_enabled': True,
                'file_path': f"./logs/{environment}.log",
                'max_file_size': '10MB',
                'backup_count': 5
            },
            'web': {
                'host': '0.0.0.0',
                'port': 8000 + (0 if environment == 'development' else 1 if environment == 'testing' else 2),
                'cors_enabled': environment != 'production',
                'debug': environment == 'development'
            },
            'monitoring': {
                'enabled': True,
                'health_check_interval': 30,
                'metrics_retention_days': 30,
                'alert_thresholds': {
                    'cpu_usage': 80,
                    'memory_usage': 85,
                    'error_rate': 5
                }
            },
            'agents': {
                'technical_analysis': {
                    'enabled': True,
                    'indicators': ['RSI', 'MACD', 'Bollinger'],
                    'timeframes': ['1m', '5m', '15m', '1h']
                },
                'risk_management': {
                    'enabled': True,
                    'max_drawdown': 15,
                    'var_confidence': 0.95
                },
                'execution': {
                    'enabled': True,
                    'slippage_tolerance': 0.1,
                    'max_retry_attempts': 3
                }
            }
        }
        
        # 环境特定覆盖
        if environment == 'production':
            base_config['logging']['level'] = 'WARNING'
            base_config['web']['debug'] = False
            base_config['monitoring']['alert_thresholds']['error_rate'] = 1
        elif environment == 'testing':
            base_config['trading']['max_position_size'] = 100.0
            base_config['database']['url'] = "sqlite:///:memory:"
        
        return base_config
    
    def _load_all_configs(self):
        """加载所有配置文件"""
        with self.lock:
            # 加载环境特定配置
            for config_file in self.base_config_dir.glob("*.yaml"):
                env_name = config_file.stem
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                    self.environment_configs[env_name] = config
                    logger.debug(f"加载配置文件: {config_file}")
                except Exception as e:
                    logger.error(f"加载配置文件失败 {config_file}: {e}")
            
            # 设置当前环境配置
            self._apply_environment_config()
    
    def _apply_environment_config(self):
        """应用当前环境配置"""
        if self.current_environment in self.environment_configs:
            self.configs = self.environment_configs[self.current_environment].copy()
            logger.info(f"应用环境配置: {self.current_environment}")
        else:
            logger.warning(f"环境配置不存在: {self.current_environment}")
    
    def _handle_file_change(self, file_path: str):
        """处理配置文件变更"""
        file_path = Path(file_path)
        if file_path.suffix in ['.yaml', '.yml'] and file_path.parent == self.base_config_dir:
            logger.info(f"检测到配置文件变更: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    new_config = yaml.safe_load(f)
                
                env_name = file_path.stem
                old_config = self.environment_configs.get(env_name, {})
                
                # 记录变更
                self._record_changes(old_config, new_config, f"file:{file_path}")
                
                # 更新配置
                with self.lock:
                    self.environment_configs[env_name] = new_config
                    if env_name == self.current_environment:
                        self._apply_environment_config()
                        self._notify_changes()
                
            except Exception as e:
                logger.error(f"处理配置文件变更失败: {e}")
    
    def _record_changes(self, old_config: Dict, new_config: Dict, source: str, prefix: str = ""):
        """记录配置变更"""
        # 扁平化比较配置
        old_flat = self._flatten_dict(old_config, prefix)
        new_flat = self._flatten_dict(new_config, prefix)
        
        all_keys = set(old_flat.keys()) | set(new_flat.keys())
        
        for key in all_keys:
            old_value = old_flat.get(key)
            new_value = new_flat.get(key)
            
            if old_value != new_value:
                change = ConfigChange(
                    timestamp=datetime.now(),
                    key=key,
                    old_value=old_value,
                    new_value=new_value,
                    source=source
                )
                self.change_history.append(change)
                
                # 保持历史记录长度
                if len(self.change_history) > 1000:
                    self.change_history = self.change_history[-500:]
    
    def _flatten_dict(self, d: Dict, prefix: str = "") -> Dict[str, Any]:
        """扁平化字典"""
        items = []
        for k, v in d.items():
            new_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _notify_changes(self):
        """通知配置变更"""
        for callback in self.change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(self.configs))
                else:
                    callback(self.configs)
            except Exception as e:
                logger.error(f"配置变更回调执行失败: {e}")
    
    def start_watching(self):
        """开始监听配置文件变更"""
        if self.file_observer is None:
            self.file_observer = Observer()
            handler = ConfigFileHandler(self)
            self.file_observer.schedule(handler, str(self.base_config_dir), recursive=False)
            self.file_observer.start()
            logger.info("开始监听配置文件变更")
    
    def stop_watching(self):
        """停止监听配置文件变更"""
        if self.file_observer:
            self.file_observer.stop()
            self.file_observer.join()
            self.file_observer = None
            logger.info("停止监听配置文件变更")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self.configs
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any, source: str = "api"):
        """设置配置值"""
        keys = key.split('.')
        
        with self.lock:
            # 记录变更
            old_value = self.get(key)
            change = ConfigChange(
                timestamp=datetime.now(),
                key=key,
                old_value=old_value,
                new_value=value,
                source=source
            )
            self.change_history.append(change)
            
            # 设置新值
            config = self.configs
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            config[keys[-1]] = value
            
            # 更新环境配置
            self.environment_configs[self.current_environment] = self.configs.copy()
            
            # 保存到文件
            self._save_environment_config(self.current_environment)
            
            # 通知变更
            self._notify_changes()
    
    def _save_environment_config(self, environment: str):
        """保存环境配置到文件"""
        try:
            config_file = self.base_config_dir / f"{environment}.yaml"
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(
                    self.environment_configs[environment], 
                    f, 
                    default_flow_style=False, 
                    allow_unicode=True
                )
            logger.debug(f"保存配置到文件: {config_file}")
        except Exception as e:
            logger.error(f"保存配置文件失败: {e}")
    
    def switch_environment(self, environment: str):
        """切换环境"""
        if environment not in self.environment_configs:
            raise ValueError(f"环境配置不存在: {environment}")
        
        old_env = self.current_environment
        self.current_environment = environment
        self._apply_environment_config()
        self._notify_changes()
        
        logger.info(f"环境切换: {old_env} -> {environment}")
    
    def add_change_callback(self, callback: Callable):
        """添加配置变更回调"""
        self.change_callbacks.append(callback)
    
    def remove_change_callback(self, callback: Callable):
        """移除配置变更回调"""
        if callback in self.change_callbacks:
            self.change_callbacks.remove(callback)
    
    def get_change_history(self, limit: int = 100) -> List[ConfigChange]:
        """获取变更历史"""
        return self.change_history[-limit:]
    
    def export_config(self, environment: str = None) -> Dict[str, Any]:
        """导出配置"""
        if environment:
            return self.environment_configs.get(environment, {}).copy()
        return self.configs.copy()
    
    def import_config(self, config: Dict[str, Any], environment: str = None):
        """导入配置"""
        env = environment or self.current_environment
        
        with self.lock:
            old_config = self.environment_configs.get(env, {})
            self._record_changes(old_config, config, f"import:{env}")
            
            self.environment_configs[env] = config.copy()
            if env == self.current_environment:
                self._apply_environment_config()
                self._notify_changes()
            
            self._save_environment_config(env)
    
    def get_status(self) -> Dict[str, Any]:
        """获取配置管理器状态"""
        return {
            "current_environment": self.current_environment,
            "available_environments": list(self.environment_configs.keys()),
            "watching": self.file_observer is not None,
            "config_keys_count": len(self._flatten_dict(self.configs)),
            "change_history_count": len(self.change_history),
            "last_change": self.change_history[-1].timestamp.isoformat() if self.change_history else None
        }


# 全局配置管理器实例
config_manager = ConfigManager()