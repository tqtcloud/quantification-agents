"""
配置管理器 (ConfigManager)
实现策略配置的动态管理、热更新和持久化
"""

import json
import yaml
import os
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
import asyncio
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from src.strategy.strategy_manager import StrategyConfig, StrategyType
from src.hft.hft_engine import HFTConfig
from src.utils.logger import LoggerMixin

# 可选导入，避免依赖问题
try:
    from src.agents.orchestrator import WorkflowConfig
except ImportError:
    WorkflowConfig = None


@dataclass
class ConfigVersion:
    """配置版本"""
    version: str
    timestamp: datetime
    description: str = ""
    author: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'version': self.version,
            'timestamp': self.timestamp.isoformat(),
            'description': self.description,
            'author': self.author
        }


@dataclass
class ConfigHistory:
    """配置历史"""
    config_id: str
    versions: List[ConfigVersion] = field(default_factory=list)
    current_version: Optional[str] = None
    
    def add_version(self, version: ConfigVersion):
        """添加新版本"""
        self.versions.append(version)
        self.current_version = version.version
    
    def get_latest_version(self) -> Optional[ConfigVersion]:
        """获取最新版本"""
        if not self.versions:
            return None
        return max(self.versions, key=lambda v: v.timestamp)


class ConfigFileWatcher(FileSystemEventHandler):
    """配置文件监视器"""
    
    def __init__(self, callback: Callable[[str], None]):
        self.callback = callback
        self.debounce_delay = 1.0  # 防抖延迟
        self.pending_events: Dict[str, asyncio.Task] = {}
    
    def on_modified(self, event):
        """文件修改事件"""
        if event.is_directory:
            return
        
        file_path = event.src_path
        if file_path.endswith(('.yaml', '.yml', '.json')):
            # 防抖处理
            if file_path in self.pending_events:
                self.pending_events[file_path].cancel()
            
            self.pending_events[file_path] = asyncio.create_task(
                self._debounced_callback(file_path)
            )
    
    async def _debounced_callback(self, file_path: str):
        """防抖回调"""
        try:
            await asyncio.sleep(self.debounce_delay)
            self.callback(file_path)
        finally:
            self.pending_events.pop(file_path, None)


class StrategyConfigManager(LoggerMixin):
    """
    策略配置管理器
    
    负责管理策略配置的加载、保存、热更新和版本控制
    """
    
    def __init__(self, config_dir: str = "config/strategies"):
        """
        初始化配置管理器
        
        Args:
            config_dir: 配置文件目录
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 配置存储
        self.configs: Dict[str, StrategyConfig] = {}
        self.config_history: Dict[str, ConfigHistory] = {}
        
        # 文件监控
        self.file_watcher: Optional[Observer] = None
        self.auto_reload: bool = True
        
        # 回调函数
        self.config_change_callbacks: List[Callable[[str, StrategyConfig], None]] = []
        
        # 线程锁
        self._config_lock = threading.RLock()
        
        self.log_info(f"配置管理器初始化完成，配置目录: {self.config_dir}")
    
    async def initialize(self):
        """初始化配置管理器"""
        try:
            # 创建配置目录结构
            await self._create_config_directories()
            
            # 加载所有配置
            await self.load_all_configs()
            
            # 启动文件监控
            if self.auto_reload:
                await self._start_file_watcher()
            
            self.log_info("配置管理器初始化完成")
            
        except Exception as e:
            self.log_error(f"配置管理器初始化失败: {e}")
            raise
    
    async def shutdown(self):
        """关闭配置管理器"""
        try:
            # 停止文件监控
            if self.file_watcher:
                self.file_watcher.stop()
                self.file_watcher.join()
            
            # 保存所有配置
            await self.save_all_configs()
            
            self.log_info("配置管理器已关闭")
            
        except Exception as e:
            self.log_error(f"关闭配置管理器失败: {e}")
    
    async def load_config(self, config_id: str) -> Optional[StrategyConfig]:
        """
        加载策略配置
        
        Args:
            config_id: 配置ID
            
        Returns:
            策略配置，如果不存在返回None
        """
        try:
            config_file = self.config_dir / f"{config_id}.yaml"
            
            if not config_file.exists():
                self.log_warning(f"配置文件不存在: {config_file}")
                return None
            
            # 读取配置文件
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # 转换为StrategyConfig对象
            config = await self._dict_to_strategy_config(config_data)
            
            if config:
                with self._config_lock:
                    self.configs[config_id] = config
                
                self.log_info(f"成功加载配置: {config_id}")
                return config
            
        except Exception as e:
            self.log_error(f"加载配置 {config_id} 失败: {e}")
            return None
    
    async def save_config(self, config: StrategyConfig, create_version: bool = True) -> bool:
        """
        保存策略配置
        
        Args:
            config: 策略配置
            create_version: 是否创建新版本
            
        Returns:
            是否保存成功
        """
        try:
            config_id = config.strategy_id
            config_file = self.config_dir / f"{config_id}.yaml"
            
            # 转换为字典
            config_dict = await self._strategy_config_to_dict(config)
            
            # 添加元数据
            config_dict['_metadata'] = {
                'created_at': datetime.now().isoformat(),
                'version': self._generate_version(),
                'config_id': config_id
            }
            
            # 保存到文件
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
            
            # 更新内存中的配置
            with self._config_lock:
                self.configs[config_id] = config
            
            # 创建版本历史
            if create_version:
                await self._create_config_version(config_id, config_dict['_metadata']['version'])
            
            self.log_info(f"配置 {config_id} 保存成功")
            return True
            
        except Exception as e:
            self.log_error(f"保存配置失败: {e}")
            return False
    
    async def load_all_configs(self):
        """加载所有配置"""
        try:
            config_files = list(self.config_dir.glob("*.yaml")) + list(self.config_dir.glob("*.yml"))
            
            for config_file in config_files:
                config_id = config_file.stem
                await self.load_config(config_id)
            
            self.log_info(f"加载了 {len(self.configs)} 个配置")
            
        except Exception as e:
            self.log_error(f"加载所有配置失败: {e}")
    
    async def save_all_configs(self):
        """保存所有配置"""
        try:
            saved_count = 0
            
            for config_id, config in self.configs.items():
                if await self.save_config(config, create_version=False):
                    saved_count += 1
            
            self.log_info(f"保存了 {saved_count} 个配置")
            
        except Exception as e:
            self.log_error(f"保存所有配置失败: {e}")
    
    def get_config(self, config_id: str) -> Optional[StrategyConfig]:
        """获取配置"""
        with self._config_lock:
            return self.configs.get(config_id)
    
    def list_configs(self, strategy_type: Optional[StrategyType] = None) -> List[Dict[str, Any]]:
        """列出所有配置"""
        configs = []
        
        with self._config_lock:
            for config_id, config in self.configs.items():
                if strategy_type is None or config.strategy_type == strategy_type:
                    configs.append({
                        'config_id': config_id,
                        'name': config.name,
                        'strategy_type': config.strategy_type.value,
                        'description': config.description,
                        'priority': config.priority,
                        'auto_restart': config.auto_restart
                    })
        
        return configs
    
    async def update_config(self, config_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新配置
        
        Args:
            config_id: 配置ID
            updates: 更新内容
            
        Returns:
            是否更新成功
        """
        try:
            with self._config_lock:
                if config_id not in self.configs:
                    self.log_error(f"配置 {config_id} 不存在")
                    return False
                
                config = self.configs[config_id]
                
                # 应用更新
                for key, value in updates.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
                    else:
                        self.log_warning(f"配置属性 {key} 不存在")
            
            # 保存更新后的配置
            await self.save_config(config)
            
            # 通知回调
            await self._notify_config_change(config_id, config)
            
            self.log_info(f"配置 {config_id} 更新成功")
            return True
            
        except Exception as e:
            self.log_error(f"更新配置 {config_id} 失败: {e}")
            return False
    
    async def delete_config(self, config_id: str) -> bool:
        """删除配置"""
        try:
            config_file = self.config_dir / f"{config_id}.yaml"
            
            # 删除文件
            if config_file.exists():
                config_file.unlink()
            
            # 从内存中删除
            with self._config_lock:
                self.configs.pop(config_id, None)
                self.config_history.pop(config_id, None)
            
            self.log_info(f"配置 {config_id} 已删除")
            return True
            
        except Exception as e:
            self.log_error(f"删除配置 {config_id} 失败: {e}")
            return False
    
    async def create_default_config(self, config_id: str, strategy_type: StrategyType) -> StrategyConfig:
        """创建默认配置"""
        try:
            # 创建基础配置
            config = StrategyConfig(
                strategy_id=config_id,
                strategy_type=strategy_type,
                name=f"{strategy_type.value.upper()} Strategy {config_id}",
                description=f"Auto-generated {strategy_type.value} strategy configuration"
            )
            
            # 根据策略类型设置特定配置
            if strategy_type == StrategyType.HFT:
                config.hft_config = HFTConfig()
                config.max_memory_mb = 2048
                config.max_cpu_percent = 50.0
                config.max_network_connections = 200
            elif strategy_type == StrategyType.AI_AGENT:
                if WorkflowConfig is not None:
                    config.workflow_config = WorkflowConfig()
                config.max_memory_mb = 1024
                config.max_cpu_percent = 25.0
                config.max_network_connections = 50
            
            # 保存配置
            await self.save_config(config)
            
            self.log_info(f"创建默认配置: {config_id}")
            return config
            
        except Exception as e:
            self.log_error(f"创建默认配置 {config_id} 失败: {e}")
            raise
    
    def register_config_change_callback(self, callback: Callable[[str, StrategyConfig], None]):
        """注册配置变更回调"""
        self.config_change_callbacks.append(callback)
    
    def get_config_history(self, config_id: str) -> Optional[List[Dict[str, Any]]]:
        """获取配置历史"""
        history = self.config_history.get(config_id)
        if not history:
            return None
        
        return [version.to_dict() for version in history.versions]
    
    async def _create_config_directories(self):
        """创建配置目录结构"""
        try:
            # 创建主配置目录
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建子目录
            (self.config_dir / "hft").mkdir(exist_ok=True)
            (self.config_dir / "ai_agent").mkdir(exist_ok=True)
            (self.config_dir / "templates").mkdir(exist_ok=True)
            (self.config_dir / "backups").mkdir(exist_ok=True)
            
            # 创建默认配置模板
            await self._create_default_templates()
            
        except Exception as e:
            self.log_error(f"创建配置目录失败: {e}")
            raise
    
    async def _create_default_templates(self):
        """创建默认配置模板"""
        try:
            templates_dir = self.config_dir / "templates"
            
            # HFT策略模板
            hft_template = {
                'strategy_id': 'hft_template',
                'strategy_type': 'hft',
                'name': 'HFT Strategy Template',
                'description': 'High Frequency Trading Strategy Template',
                'max_memory_mb': 2048,
                'max_cpu_percent': 50.0,
                'max_network_connections': 200,
                'priority': 2,
                'auto_restart': True,
                'max_restarts': 5,
                'restart_delay_seconds': 30,
                'health_check_interval': 10,
                'hft_config': {
                    'max_orderbook_levels': 50,
                    'orderbook_history_size': 1000,
                    'microstructure_lookback': 100,
                    'min_signal_strength': 0.3,
                    'max_orders': 10000,
                    'update_interval_ms': 1.0,
                    'latency_target_ms': 10.0
                }
            }
            
            # AI Agent策略模板
            ai_template = {
                'strategy_id': 'ai_agent_template',
                'strategy_type': 'ai_agent',
                'name': 'AI Agent Strategy Template',
                'description': 'AI Agent Strategy Template',
                'max_memory_mb': 1024,
                'max_cpu_percent': 25.0,
                'max_network_connections': 50,
                'priority': 1,
                'auto_restart': True,
                'max_restarts': 3,
                'restart_delay_seconds': 60,
                'health_check_interval': 30
            }
            
            # 只有在WorkflowConfig可用时才添加工作流配置
            if WorkflowConfig is not None:
                ai_template['workflow_config'] = {
                    'max_parallel_agents': 6,
                    'enable_checkpointing': True,
                    'checkpoint_interval': 5,
                    'timeout_seconds': 300,
                    'retry_failed_nodes': True,
                    'max_retries': 3,
                    'aggregation_method': 'weighted_voting',
                    'consensus_threshold': 0.6
                }
            
            # 保存模板文件
            for template, filename in [(hft_template, 'hft_template.yaml'), (ai_template, 'ai_agent_template.yaml')]:
                template_file = templates_dir / filename
                if not template_file.exists():
                    with open(template_file, 'w', encoding='utf-8') as f:
                        yaml.dump(template, f, default_flow_style=False, allow_unicode=True)
            
        except Exception as e:
            self.log_error(f"创建默认模板失败: {e}")
    
    async def _start_file_watcher(self):
        """启动文件监控"""
        try:
            event_handler = ConfigFileWatcher(self._on_config_file_changed)
            self.file_watcher = Observer()
            self.file_watcher.schedule(event_handler, str(self.config_dir), recursive=True)
            self.file_watcher.start()
            
            self.log_info("配置文件监控已启动")
            
        except Exception as e:
            self.log_error(f"启动文件监控失败: {e}")
    
    def _on_config_file_changed(self, file_path: str):
        """配置文件变更回调"""
        try:
            config_id = Path(file_path).stem
            
            # 重新加载配置
            asyncio.create_task(self._reload_config(config_id))
            
        except Exception as e:
            self.log_error(f"处理配置文件变更失败: {e}")
    
    async def _reload_config(self, config_id: str):
        """重新加载配置"""
        try:
            old_config = self.configs.get(config_id)
            new_config = await self.load_config(config_id)
            
            if new_config and old_config != new_config:
                await self._notify_config_change(config_id, new_config)
                self.log_info(f"配置 {config_id} 已重新加载")
            
        except Exception as e:
            self.log_error(f"重新加载配置 {config_id} 失败: {e}")
    
    async def _notify_config_change(self, config_id: str, config: StrategyConfig):
        """通知配置变更"""
        try:
            for callback in self.config_change_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(config_id, config)
                    else:
                        callback(config_id, config)
                except Exception as e:
                    self.log_error(f"配置变更回调执行失败: {e}")
        except Exception as e:
            self.log_error(f"通知配置变更失败: {e}")
    
    async def _dict_to_strategy_config(self, config_data: Dict[str, Any]) -> Optional[StrategyConfig]:
        """将字典转换为StrategyConfig对象"""
        try:
            # 基础配置
            config = StrategyConfig(
                strategy_id=config_data['strategy_id'],
                strategy_type=StrategyType(config_data['strategy_type']),
                name=config_data['name'],
                description=config_data.get('description', ''),
                max_memory_mb=config_data.get('max_memory_mb', 1024),
                max_cpu_percent=config_data.get('max_cpu_percent', 25.0),
                max_network_connections=config_data.get('max_network_connections', 100),
                priority=config_data.get('priority', 1),
                auto_restart=config_data.get('auto_restart', True),
                max_restarts=config_data.get('max_restarts', 5),
                restart_delay_seconds=config_data.get('restart_delay_seconds', 30),
                health_check_interval=config_data.get('health_check_interval', 10)
            )
            
            # HFT特定配置
            if config.strategy_type == StrategyType.HFT and 'hft_config' in config_data:
                hft_data = config_data['hft_config']
                config.hft_config = HFTConfig(
                    max_orderbook_levels=hft_data.get('max_orderbook_levels', 50),
                    orderbook_history_size=hft_data.get('orderbook_history_size', 1000),
                    microstructure_lookback=hft_data.get('microstructure_lookback', 100),
                    min_signal_strength=hft_data.get('min_signal_strength', 0.3),
                    max_orders=hft_data.get('max_orders', 10000),
                    update_interval_ms=hft_data.get('update_interval_ms', 1.0),
                    latency_target_ms=hft_data.get('latency_target_ms', 10.0)
                )
            
            # AI Agent特定配置
            elif config.strategy_type == StrategyType.AI_AGENT and 'workflow_config' in config_data:
                if WorkflowConfig is not None:
                    workflow_data = config_data['workflow_config']
                    config.workflow_config = WorkflowConfig(
                        max_parallel_agents=workflow_data.get('max_parallel_agents', 6),
                        enable_checkpointing=workflow_data.get('enable_checkpointing', True),
                        checkpoint_interval=workflow_data.get('checkpoint_interval', 5),
                        timeout_seconds=workflow_data.get('timeout_seconds', 300),
                        retry_failed_nodes=workflow_data.get('retry_failed_nodes', True),
                        max_retries=workflow_data.get('max_retries', 3),
                        aggregation_method=workflow_data.get('aggregation_method', 'weighted_voting'),
                        consensus_threshold=workflow_data.get('consensus_threshold', 0.6)
                    )
            
            return config
            
        except Exception as e:
            self.log_error(f"转换配置失败: {e}")
            return None
    
    async def _strategy_config_to_dict(self, config: StrategyConfig) -> Dict[str, Any]:
        """将StrategyConfig对象转换为字典"""
        try:
            config_dict = {
                'strategy_id': config.strategy_id,
                'strategy_type': config.strategy_type.value,
                'name': config.name,
                'description': config.description,
                'max_memory_mb': config.max_memory_mb,
                'max_cpu_percent': config.max_cpu_percent,
                'max_network_connections': config.max_network_connections,
                'priority': config.priority,
                'auto_restart': config.auto_restart,
                'max_restarts': config.max_restarts,
                'restart_delay_seconds': config.restart_delay_seconds,
                'health_check_interval': config.health_check_interval
            }
            
            # HFT特定配置
            if config.hft_config:
                config_dict['hft_config'] = asdict(config.hft_config)
            
            # AI Agent特定配置
            if config.workflow_config:
                config_dict['workflow_config'] = asdict(config.workflow_config)
            
            return config_dict
            
        except Exception as e:
            self.log_error(f"转换配置为字典失败: {e}")
            return {}
    
    def _generate_version(self) -> str:
        """生成版本号"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    async def _create_config_version(self, config_id: str, version: str):
        """创建配置版本"""
        try:
            if config_id not in self.config_history:
                self.config_history[config_id] = ConfigHistory(config_id)
            
            history = self.config_history[config_id]
            version_info = ConfigVersion(
                version=version,
                timestamp=datetime.now(),
                description=f"Auto-generated version {version}"
            )
            
            history.add_version(version_info)
            
        except Exception as e:
            self.log_error(f"创建配置版本失败: {e}")