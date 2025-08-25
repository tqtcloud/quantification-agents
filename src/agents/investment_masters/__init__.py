"""
投资大师Agent集合
包含15个专业投资分析师Agent
"""

from .warren_buffett import WarrenBuffettAgent
from .cathie_wood import CathieWoodAgent
from .ray_dalio import RayDalioAgent
from .benjamin_graham import BenjaminGrahamAgent
# from .peter_lynch import PeterLynchAgent
# from .charlie_munger import CharlieMungerAgent
# from .philip_fisher import PhilipFisherAgent
# from .george_soros import GeorgeSorosAgent
from .technical_analyst import TechnicalAnalystAgent
from .quantitative_analyst import QuantitativeAnalystAgent
# from .macro_economist import MacroEconomistAgent
# from .sector_analyst import SectorAnalystAgent
# from .esg_analyst import ESGAnalystAgent
# from .crypto_specialist import CryptoSpecialistAgent
# from .options_strategist import OptionsStrategistAgent

__all__ = [
    # 价值投资类
    "WarrenBuffettAgent",
    "BenjaminGrahamAgent",
    # "PeterLynchAgent",
    # "CharlieMungerAgent",
    
    # 成长投资类
    "CathieWoodAgent",
    # "PhilipFisherAgent",
    
    # 宏观策略类
    "RayDalioAgent",
    # "GeorgeSorosAgent",
    
    # 专业分析师类
    "TechnicalAnalystAgent",
    "QuantitativeAnalystAgent",
    # "MacroEconomistAgent",
    # "SectorAnalystAgent",
    # "ESGAnalystAgent",
    # "CryptoSpecialistAgent",
    # "OptionsStrategistAgent",
]

# Agent类型映射
AGENT_REGISTRY = {
    "warren_buffett": WarrenBuffettAgent,
    "cathie_wood": CathieWoodAgent,
    "ray_dalio": RayDalioAgent,
    "benjamin_graham": BenjaminGrahamAgent,
    # "peter_lynch": PeterLynchAgent,
    # "charlie_munger": CharlieMungerAgent,
    # "philip_fisher": PhilipFisherAgent,
    # "george_soros": GeorgeSorosAgent,
    "technical_analyst": TechnicalAnalystAgent,
    "quantitative_analyst": QuantitativeAnalystAgent,
    # "macro_economist": MacroEconomistAgent,
    # "sector_analyst": SectorAnalystAgent,
    # "esg_analyst": ESGAnalystAgent,
    # "crypto_specialist": CryptoSpecialistAgent,
    # "options_strategist": OptionsStrategistAgent,
}

def get_agent_by_name(name: str, config=None, message_bus=None):
    """根据名称获取Agent实例"""
    agent_class = AGENT_REGISTRY.get(name.lower())
    if agent_class:
        return agent_class(config=config, message_bus=message_bus)
    else:
        raise ValueError(f"Unknown agent name: {name}")

def list_available_agents():
    """列出所有可用的Agent"""
    return list(AGENT_REGISTRY.keys())