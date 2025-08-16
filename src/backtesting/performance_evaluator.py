import json
import statistics
import numpy as np
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from src.backtesting.backtest_engine import BacktestResult
from src.trading.performance_analytics import PerformanceMetrics, TradeRecord
from src.utils.logger import LoggerMixin


@dataclass
class BacktestComparison:
    """回测比较结果"""
    strategy_names: List[str]
    comparison_metrics: Dict[str, Dict[str, float]]
    relative_performance: Dict[str, float]
    correlation_matrix: Dict[str, Dict[str, float]]
    risk_adjusted_ranking: List[Tuple[str, float]]  # (strategy_name, score)


@dataclass 
class BenchmarkComparison:
    """基准比较结果"""
    benchmark_name: str
    benchmark_return: float
    strategy_return: float
    alpha: float
    beta: float
    information_ratio: float
    tracking_error: float
    up_capture: float
    down_capture: float


@dataclass
class RiskAnalysis:
    """风险分析结果"""
    var_95: float
    var_99: float
    cvar_95: float  # 条件风险价值
    max_drawdown: float
    max_drawdown_duration: int  # 天数
    volatility: float
    skewness: float
    kurtosis: float
    tail_ratio: float  # 95%分位数 / 5%分位数


class PerformanceEvaluator(LoggerMixin):
    """回测性能评估器"""
    
    def __init__(self):
        self.results_cache: Dict[str, BacktestResult] = {}
        
    def add_backtest_result(self, name: str, result: BacktestResult):
        """添加回测结果"""
        self.results_cache[name] = result
        self.log_info(f"Added backtest result: {name}")
    
    def evaluate_single_strategy(self, result: BacktestResult) -> Dict[str, Any]:
        """评估单个策略性能"""
        self.log_info("Evaluating single strategy performance...")
        
        # 基础性能指标
        metrics = result.performance_metrics
        
        # 高级风险分析
        risk_analysis = self._calculate_risk_metrics(result)
        
        # 时间序列分析
        time_analysis = self._analyze_time_patterns(result)
        
        # 交易质量分析
        trade_analysis = self._analyze_trade_quality(result)
        
        evaluation = {
            "basic_metrics": metrics,
            "risk_analysis": asdict(risk_analysis),
            "time_analysis": time_analysis,
            "trade_analysis": trade_analysis,
            "summary_score": self._calculate_summary_score(metrics, risk_analysis)
        }
        
        return evaluation
    
    def compare_strategies(self, strategy_names: Optional[List[str]] = None) -> BacktestComparison:
        """比较多个策略"""
        if strategy_names is None:
            strategy_names = list(self.results_cache.keys())
        
        if len(strategy_names) < 2:
            raise ValueError("Need at least 2 strategies for comparison")
        
        self.log_info(f"Comparing strategies: {strategy_names}")
        
        # 收集比较指标
        comparison_metrics = {}
        returns_series = {}
        
        for name in strategy_names:
            if name not in self.results_cache:
                self.log_warning(f"Strategy {name} not found in cache")
                continue
                
            result = self.results_cache[name]
            metrics = result.performance_metrics
            
            comparison_metrics[name] = {
                "total_return": metrics.get("total_return", 0),
                "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                "max_drawdown": metrics.get("max_drawdown", 0),
                "win_rate": metrics.get("win_rate", 0),
                "profit_factor": metrics.get("profit_factor", 0),
                "calmar_ratio": metrics.get("calmar_ratio", 0),
                "sortino_ratio": metrics.get("sortino_ratio", 0)
            }
            
            # 获取收益率序列
            returns_series[name] = self._extract_returns_series(result)
        
        # 计算相对性能
        relative_performance = self._calculate_relative_performance(comparison_metrics)
        
        # 计算相关系数矩阵
        correlation_matrix = self._calculate_correlation_matrix(returns_series)
        
        # 风险调整排名
        risk_adjusted_ranking = self._calculate_risk_adjusted_ranking(comparison_metrics)
        
        return BacktestComparison(
            strategy_names=strategy_names,
            comparison_metrics=comparison_metrics,
            relative_performance=relative_performance,
            correlation_matrix=correlation_matrix,
            risk_adjusted_ranking=risk_adjusted_ranking
        )
    
    def compare_to_benchmark(self, strategy_name: str, benchmark_returns: List[float],
                           benchmark_name: str = "Benchmark") -> BenchmarkComparison:
        """与基准比较"""
        if strategy_name not in self.results_cache:
            raise ValueError(f"Strategy {strategy_name} not found")
        
        result = self.results_cache[strategy_name]
        strategy_returns = self._extract_returns_series(result)
        
        if len(strategy_returns) != len(benchmark_returns):
            self.log_warning("Strategy and benchmark returns length mismatch")
            # 截取到较短的长度
            min_length = min(len(strategy_returns), len(benchmark_returns))
            strategy_returns = strategy_returns[:min_length]
            benchmark_returns = benchmark_returns[:min_length]
        
        # 计算基本指标
        strategy_return = sum(strategy_returns)
        benchmark_return = sum(benchmark_returns)
        
        # 计算贝塔和阿尔法
        if len(strategy_returns) > 1 and len(benchmark_returns) > 1:
            covariance = np.cov(strategy_returns, benchmark_returns)[0][1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
            
            # 阿尔法 = 策略收益 - (贝塔 * 基准收益)
            alpha = strategy_return - (beta * benchmark_return)
            
            # 信息比率和跟踪误差
            excess_returns = [s - b for s, b in zip(strategy_returns, benchmark_returns)]
            tracking_error = np.std(excess_returns) if len(excess_returns) > 1 else 0
            information_ratio = np.mean(excess_returns) / tracking_error if tracking_error != 0 else 0
            
        else:
            beta = 0
            alpha = strategy_return - benchmark_return
            information_ratio = 0
            tracking_error = 0
        
        # 上涨/下跌捕获率
        up_periods = [i for i, b in enumerate(benchmark_returns) if b > 0]
        down_periods = [i for i, b in enumerate(benchmark_returns) if b < 0]
        
        if up_periods:
            up_strategy = sum(strategy_returns[i] for i in up_periods)
            up_benchmark = sum(benchmark_returns[i] for i in up_periods)
            up_capture = up_strategy / up_benchmark if up_benchmark != 0 else 0
        else:
            up_capture = 0
            
        if down_periods:
            down_strategy = sum(strategy_returns[i] for i in down_periods)
            down_benchmark = sum(benchmark_returns[i] for i in down_periods)
            down_capture = down_strategy / down_benchmark if down_benchmark != 0 else 0
        else:
            down_capture = 0
        
        return BenchmarkComparison(
            benchmark_name=benchmark_name,
            benchmark_return=benchmark_return,
            strategy_return=strategy_return,
            alpha=alpha,
            beta=beta,
            information_ratio=information_ratio,
            tracking_error=tracking_error,
            up_capture=up_capture,
            down_capture=down_capture
        )
    
    def _calculate_risk_metrics(self, result: BacktestResult) -> RiskAnalysis:
        """计算高级风险指标"""
        returns = self._extract_returns_series(result)
        
        if len(returns) < 10:
            # 数据不足，返回默认值
            return RiskAnalysis(
                var_95=0, var_99=0, cvar_95=0, max_drawdown=0,
                max_drawdown_duration=0, volatility=0, skewness=0,
                kurtosis=0, tail_ratio=1
            )
        
        # VaR计算
        sorted_returns = sorted(returns)
        var_95 = -sorted_returns[int(len(returns) * 0.05)]
        var_99 = -sorted_returns[int(len(returns) * 0.01)]
        
        # CVaR计算（条件风险价值）
        var_95_threshold = sorted_returns[int(len(returns) * 0.05)]
        tail_losses = [r for r in sorted_returns if r <= var_95_threshold]
        cvar_95 = -np.mean(tail_losses) if tail_losses else 0
        
        # 波动率
        volatility = np.std(returns) * np.sqrt(252)  # 年化波动率
        
        # 偏度和峰度
        skewness = float(statistics.stdev(returns) ** 3)
        kurtosis = self._calculate_kurtosis(returns)
        
        # 尾部比率
        p95 = np.percentile(returns, 95)
        p5 = np.percentile(returns, 5)
        tail_ratio = p95 / abs(p5) if p5 != 0 else 1
        
        # 最大回撤持续时间
        max_dd_duration = self._calculate_max_drawdown_duration(result)
        
        return RiskAnalysis(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            max_drawdown=result.performance_metrics.get("max_drawdown", 0),
            max_drawdown_duration=max_dd_duration,
            volatility=volatility,
            skewness=skewness,
            kurtosis=kurtosis,
            tail_ratio=tail_ratio
        )
    
    def _calculate_kurtosis(self, returns: List[float]) -> float:
        """计算峰度"""
        if len(returns) < 4:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        normalized = [(r - mean_return) / std_return for r in returns]
        kurtosis = np.mean([x**4 for x in normalized]) - 3  # 减去3得到超额峰度
        
        return kurtosis
    
    def _calculate_max_drawdown_duration(self, result: BacktestResult) -> int:
        """计算最大回撤持续时间（天数）"""
        equity_curve = result.equity_curve
        
        if len(equity_curve) < 2:
            return 0
        
        peak = equity_curve[0][1]
        max_duration = 0
        current_duration = 0
        
        for timestamp, equity in equity_curve:
            if equity > peak:
                peak = equity
                current_duration = 0
            else:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
        
        return max_duration
    
    def _analyze_time_patterns(self, result: BacktestResult) -> Dict[str, Any]:
        """分析时间模式"""
        trades = result.trade_records
        
        if not trades:
            return {"monthly_returns": {}, "weekday_analysis": {}, "hourly_analysis": {}}
        
        # 按月份分析收益
        monthly_returns = {}
        for trade in trades:
            if trade.exit_time:
                month = trade.exit_time.strftime("%Y-%m")
                if month not in monthly_returns:
                    monthly_returns[month] = 0
                monthly_returns[month] += float(trade.net_pnl)
        
        # 按星期几分析
        weekday_pnl = {i: [] for i in range(7)}  # 0=Monday, 6=Sunday
        for trade in trades:
            if trade.exit_time:
                weekday = trade.exit_time.weekday()
                weekday_pnl[weekday].append(float(trade.net_pnl))
        
        weekday_analysis = {}
        for day, pnls in weekday_pnl.items():
            if pnls:
                weekday_analysis[day] = {
                    "avg_pnl": np.mean(pnls),
                    "win_rate": len([p for p in pnls if p > 0]) / len(pnls),
                    "trade_count": len(pnls)
                }
        
        # 按小时分析（如果有足够的数据）
        hourly_pnl = {i: [] for i in range(24)}
        for trade in trades:
            if trade.exit_time:
                hour = trade.exit_time.hour
                hourly_pnl[hour].append(float(trade.net_pnl))
        
        hourly_analysis = {}
        for hour, pnls in hourly_pnl.items():
            if len(pnls) >= 5:  # 至少5笔交易才分析
                hourly_analysis[hour] = {
                    "avg_pnl": np.mean(pnls),
                    "win_rate": len([p for p in pnls if p > 0]) / len(pnls),
                    "trade_count": len(pnls)
                }
        
        return {
            "monthly_returns": monthly_returns,
            "weekday_analysis": weekday_analysis,
            "hourly_analysis": hourly_analysis
        }
    
    def _analyze_trade_quality(self, result: BacktestResult) -> Dict[str, Any]:
        """分析交易质量"""
        trades = result.trade_records
        
        if not trades:
            return {}
        
        # 计算交易持续时间分析
        durations = []
        for trade in trades:
            duration = trade.get_duration()
            if duration:
                durations.append(duration.total_seconds() / 3600)  # 转为小时
        
        # 连续盈亏分析
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_streak = 0
        current_streak_type = None
        
        for trade in sorted(trades, key=lambda x: x.entry_time):
            if trade.net_pnl > 0:
                if current_streak_type == "win":
                    current_streak += 1
                else:
                    current_streak = 1
                    current_streak_type = "win"
                max_consecutive_wins = max(max_consecutive_wins, current_streak)
            elif trade.net_pnl < 0:
                if current_streak_type == "loss":
                    current_streak += 1
                else:
                    current_streak = 1
                    current_streak_type = "loss"
                max_consecutive_losses = max(max_consecutive_losses, current_streak)
        
        # 最大单笔盈利/亏损
        pnls = [float(trade.net_pnl) for trade in trades]
        max_win = max(pnls) if pnls else 0
        max_loss = min(pnls) if pnls else 0
        
        return {
            "trade_count": len(trades),
            "avg_duration_hours": np.mean(durations) if durations else 0,
            "max_duration_hours": max(durations) if durations else 0,
            "min_duration_hours": min(durations) if durations else 0,
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses,
            "max_single_win": max_win,
            "max_single_loss": max_loss,
            "large_win_ratio": len([p for p in pnls if p > max_win * 0.5]) / len(pnls) if pnls else 0,
            "large_loss_ratio": len([p for p in pnls if p < max_loss * 0.5]) / len(pnls) if pnls else 0
        }
    
    def _extract_returns_series(self, result: BacktestResult) -> List[float]:
        """提取收益率序列"""
        if not result.trade_records:
            return []
        
        # 按交易计算收益率
        returns = []
        for trade in sorted(result.trade_records, key=lambda x: x.entry_time):
            return_rate = trade.get_return_rate()
            returns.append(float(return_rate))
        
        return returns
    
    def _calculate_relative_performance(self, metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """计算相对性能"""
        if not metrics:
            return {}
        
        # 计算每个指标的平均值
        avg_metrics = {}
        for metric_name in next(iter(metrics.values())).keys():
            values = [strategy_metrics[metric_name] for strategy_metrics in metrics.values()]
            avg_metrics[metric_name] = np.mean(values)
        
        # 计算相对性能分数
        relative_scores = {}
        for strategy_name, strategy_metrics in metrics.items():
            score = 0
            for metric_name, value in strategy_metrics.items():
                if avg_metrics[metric_name] != 0:
                    # 对于负面指标（如最大回撤），反转比较
                    if metric_name in ["max_drawdown"]:
                        relative_score = avg_metrics[metric_name] / value if value != 0 else 1
                    else:
                        relative_score = value / avg_metrics[metric_name]
                    score += relative_score
            
            relative_scores[strategy_name] = score / len(strategy_metrics)
        
        return relative_scores
    
    def _calculate_correlation_matrix(self, returns_series: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """计算收益率相关系数矩阵"""
        correlation_matrix = {}
        
        strategies = list(returns_series.keys())
        for i, strategy1 in enumerate(strategies):
            correlation_matrix[strategy1] = {}
            for j, strategy2 in enumerate(strategies):
                if i == j:
                    correlation_matrix[strategy1][strategy2] = 1.0
                elif len(returns_series[strategy1]) > 1 and len(returns_series[strategy2]) > 1:
                    # 确保两个序列长度相同
                    min_length = min(len(returns_series[strategy1]), len(returns_series[strategy2]))
                    returns1 = returns_series[strategy1][:min_length]
                    returns2 = returns_series[strategy2][:min_length]
                    
                    correlation = np.corrcoef(returns1, returns2)[0, 1]
                    correlation_matrix[strategy1][strategy2] = correlation if not np.isnan(correlation) else 0
                else:
                    correlation_matrix[strategy1][strategy2] = 0
        
        return correlation_matrix
    
    def _calculate_risk_adjusted_ranking(self, metrics: Dict[str, Dict[str, float]]) -> List[Tuple[str, float]]:
        """计算风险调整排名"""
        scores = {}
        
        for strategy_name, strategy_metrics in metrics.items():
            # 风险调整得分 = (夏普比率 * 0.4 + 卡尔玛比率 * 0.3 + 索提诺比率 * 0.3)
            sharpe = strategy_metrics.get("sharpe_ratio", 0)
            calmar = strategy_metrics.get("calmar_ratio", 0)
            sortino = strategy_metrics.get("sortino_ratio", 0)
            
            risk_adjusted_score = sharpe * 0.4 + calmar * 0.3 + sortino * 0.3
            scores[strategy_name] = risk_adjusted_score
        
        # 按得分排序
        ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranking
    
    def _calculate_summary_score(self, metrics: Dict[str, Any], risk_analysis: RiskAnalysis) -> float:
        """计算综合评分"""
        # 综合评分公式（0-100分）
        score = 0
        
        # 收益率部分（30%）
        total_return = metrics.get("total_return", 0)
        return_score = min(total_return * 100, 30) if total_return > 0 else max(total_return * 100, -30)
        score += return_score * 0.3
        
        # 夏普比率部分（25%）
        sharpe = metrics.get("sharpe_ratio", 0)
        sharpe_score = min(sharpe * 15, 25) if sharpe > 0 else max(sharpe * 15, -25)
        score += sharpe_score * 0.25
        
        # 最大回撤部分（25%）
        max_dd = abs(metrics.get("max_drawdown", 0))
        dd_score = max(25 - max_dd * 100, 0)  # 回撤越小得分越高
        score += dd_score * 0.25
        
        # 胜率部分（20%）
        win_rate = metrics.get("win_rate", 0)
        win_score = win_rate * 20
        score += win_score * 0.2
        
        return max(min(score, 100), 0)  # 限制在0-100范围内
    
    def generate_report(self, strategy_name: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """生成详细的性能报告"""
        if strategy_name not in self.results_cache:
            raise ValueError(f"Strategy {strategy_name} not found")
        
        result = self.results_cache[strategy_name]
        evaluation = self.evaluate_single_strategy(result)
        
        # 生成报告
        report = {
            "strategy_name": strategy_name,
            "report_timestamp": datetime.utcnow().isoformat(),
            "backtest_period": {
                "start_date": result.config.start_date.isoformat(),
                "end_date": result.config.end_date.isoformat(),
                "duration_days": (result.config.end_date - result.config.start_date).days
            },
            "performance_evaluation": evaluation,
            "config": asdict(result.config),
            "execution_summary": {
                "total_events": result.total_events,
                "processed_events": result.processed_events,
                "execution_errors_count": len(result.execution_errors),
                "duration_seconds": result.duration_seconds
            }
        }
        
        # 保存报告
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            self.log_info(f"Report saved to {output_path}")
        
        return report