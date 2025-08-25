"""
测试报告生成器
生成详细的策略管理系统测试报告
"""

import json
import time
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import xml.etree.ElementTree as ET
import statistics


@dataclass
class TestResult:
    """测试结果数据类"""
    test_name: str
    test_class: str
    status: str  # passed, failed, skipped, error
    duration: float
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    setup_duration: float = 0.0
    teardown_duration: float = 0.0
    
    @property
    def total_duration(self) -> float:
        return self.duration + self.setup_duration + self.teardown_duration


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    latency_avg_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_max_ms: float = 0.0
    throughput_avg_tps: float = 0.0
    throughput_max_tps: float = 0.0
    memory_avg_mb: float = 0.0
    memory_max_mb: float = 0.0
    cpu_avg_percent: float = 0.0
    cpu_max_percent: float = 0.0
    test_duration_seconds: float = 0.0
    
    def meets_targets(self, targets: Dict[str, float]) -> Dict[str, bool]:
        """检查是否满足性能目标"""
        results = {}
        
        if 'latency_target_ms' in targets:
            results['latency'] = self.latency_avg_ms <= targets['latency_target_ms']
        
        if 'throughput_target_tps' in targets:
            results['throughput'] = self.throughput_avg_tps >= targets['throughput_target_tps']
            
        if 'memory_target_mb' in targets:
            results['memory'] = self.memory_max_mb <= targets['memory_target_mb']
            
        if 'cpu_target_percent' in targets:
            results['cpu'] = self.cpu_max_percent <= targets['cpu_target_percent']
            
        return results


@dataclass
class CoverageInfo:
    """代码覆盖率信息"""
    line_coverage: float = 0.0
    branch_coverage: float = 0.0
    function_coverage: float = 0.0
    total_lines: int = 0
    covered_lines: int = 0
    missing_lines: List[str] = field(default_factory=list)
    
    @property
    def coverage_grade(self) -> str:
        """覆盖率等级"""
        if self.line_coverage >= 0.95:
            return "A+"
        elif self.line_coverage >= 0.9:
            return "A"
        elif self.line_coverage >= 0.8:
            return "B"
        elif self.line_coverage >= 0.7:
            return "C"
        else:
            return "D"


@dataclass
class TestSummary:
    """测试摘要"""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    error_tests: int = 0
    total_duration: float = 0.0
    test_results: List[TestResult] = field(default_factory=list)
    performance_metrics: Optional[PerformanceMetrics] = None
    coverage_info: Optional[CoverageInfo] = None
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests
    
    @property
    def failure_rate(self) -> float:
        """失败率"""
        if self.total_tests == 0:
            return 0.0
        return (self.failed_tests + self.error_tests) / self.total_tests


class TestReportGenerator:
    """测试报告生成器"""
    
    def __init__(self, output_dir: str = "logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 性能目标
        self.performance_targets = {
            'latency_target_ms': 10.0,
            'throughput_target_tps': 1000.0,
            'memory_target_mb': 512.0,
            'cpu_target_percent': 80.0
        }
    
    def parse_pytest_json_report(self, json_file: str) -> TestSummary:
        """解析pytest JSON报告"""
        summary = TestSummary()
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 解析测试摘要
            test_summary = data.get('summary', {})
            summary.total_tests = test_summary.get('total', 0)
            summary.passed_tests = test_summary.get('passed', 0)
            summary.failed_tests = test_summary.get('failed', 0)
            summary.skipped_tests = test_summary.get('skipped', 0)
            summary.error_tests = test_summary.get('error', 0)
            summary.total_duration = test_summary.get('duration', 0.0)
            
            # 解析测试详情
            for test in data.get('tests', []):
                result = TestResult(
                    test_name=test.get('nodeid', '').split('::')[-1],
                    test_class=test.get('nodeid', '').split('::')[1] if '::' in test.get('nodeid', '') else '',
                    status=test.get('outcome', 'unknown'),
                    duration=test.get('duration', 0.0)
                )
                
                # 错误信息
                if test.get('call', {}).get('longrepr'):
                    result.error_message = str(test['call']['longrepr'])
                
                summary.test_results.append(result)
                
        except Exception as e:
            print(f"解析pytest JSON报告时出错: {e}")
        
        return summary
    
    def parse_coverage_xml_report(self, xml_file: str) -> CoverageInfo:
        """解析覆盖率XML报告"""
        coverage = CoverageInfo()
        
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            coverage.line_coverage = float(root.get('line-rate', 0))
            coverage.branch_coverage = float(root.get('branch-rate', 0))
            
            # 解析详细覆盖信息
            total_lines = 0
            covered_lines = 0
            
            for package in root.findall('.//package'):
                for class_elem in package.findall('.//class'):
                    for line in class_elem.findall('.//line'):
                        total_lines += 1
                        if int(line.get('hits', 0)) > 0:
                            covered_lines += 1
            
            coverage.total_lines = total_lines
            coverage.covered_lines = covered_lines
            
        except Exception as e:
            print(f"解析覆盖率XML报告时出错: {e}")
        
        return coverage
    
    def extract_performance_metrics(self, test_results: List[TestResult]) -> PerformanceMetrics:
        """从测试结果中提取性能指标"""
        metrics = PerformanceMetrics()
        
        # 查找性能相关的测试
        perf_tests = [r for r in test_results if 'performance' in r.test_name.lower() or 'benchmark' in r.test_name.lower()]
        
        if perf_tests:
            durations = [t.duration for t in perf_tests]
            metrics.latency_avg_ms = statistics.mean(durations) * 1000  # 转换为毫秒
            metrics.latency_p95_ms = statistics.quantiles(durations, n=20)[18] * 1000 if len(durations) >= 20 else 0
            metrics.latency_p99_ms = statistics.quantiles(durations, n=100)[98] * 1000 if len(durations) >= 100 else 0
            metrics.latency_max_ms = max(durations) * 1000
            metrics.test_duration_seconds = sum(durations)
        
        return metrics
    
    def generate_html_report(self, summary: TestSummary, output_file: str = None) -> str:
        """生成HTML测试报告"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"strategy_test_report_{timestamp}.html"
        
        # HTML模板
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>双策略管理系统测试报告</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .header {{
            text-align: center;
            border-bottom: 2px solid #007ACC;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            color: #007ACC;
            margin: 0;
            font-size: 2.5em;
        }}
        
        .header p {{
            color: #666;
            margin: 10px 0 0 0;
            font-size: 1.1em;
        }}
        
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        
        .card.success {{
            background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        }}
        
        .card.warning {{
            background: linear-gradient(135deg, #FF9800 0%, #f57c00 100%);
        }}
        
        .card.error {{
            background: linear-gradient(135deg, #f44336 0%, #d32f2f 100%);
        }}
        
        .card h3 {{
            margin: 0 0 10px 0;
            font-size: 2em;
        }}
        
        .card p {{
            margin: 0;
            opacity: 0.9;
        }}
        
        .section {{
            margin-bottom: 30px;
            background: #fafafa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #007ACC;
        }}
        
        .section h2 {{
            color: #333;
            margin: 0 0 20px 0;
            font-size: 1.5em;
        }}
        
        .performance-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }}
        
        .metric {{
            background: white;
            padding: 15px;
            border-radius: 6px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        .metric-name {{
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
        }}
        
        .metric-value {{
            font-size: 1.2em;
            color: #007ACC;
        }}
        
        .metric-status {{
            display: inline-block;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: bold;
            margin-left: 10px;
        }}
        
        .status-pass {{
            background: #4CAF50;
            color: white;
        }}
        
        .status-fail {{
            background: #f44336;
            color: white;
        }}
        
        .test-results {{
            background: white;
            border-radius: 6px;
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        .test-result {{
            padding: 15px;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        
        .test-result:last-child {{
            border-bottom: none;
        }}
        
        .test-name {{
            font-weight: bold;
            color: #333;
        }}
        
        .test-class {{
            color: #666;
            font-size: 0.9em;
        }}
        
        .test-duration {{
            color: #007ACC;
            font-size: 0.9em;
        }}
        
        .coverage-bar {{
            width: 100%;
            height: 20px;
            background: #eee;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }}
        
        .coverage-fill {{
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #45a049);
            transition: width 0.3s ease;
        }}
        
        .coverage-text {{
            text-align: center;
            margin-top: 5px;
            font-weight: bold;
        }}
        
        .grade {{
            display: inline-block;
            padding: 5px 10px;
            border-radius: 15px;
            font-weight: bold;
            margin-left: 10px;
        }}
        
        .grade-a {{
            background: #4CAF50;
            color: white;
        }}
        
        .grade-b {{
            background: #FF9800;
            color: white;
        }}
        
        .grade-c {{
            background: #f44336;
            color: white;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #eee;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 双策略管理系统测试报告</h1>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="summary-cards">
            <div class="card success">
                <h3>{summary.total_tests}</h3>
                <p>总测试数</p>
            </div>
            <div class="card {'success' if summary.passed_tests == summary.total_tests else 'warning'}">
                <h3>{summary.passed_tests}</h3>
                <p>通过测试</p>
            </div>
            <div class="card {'error' if summary.failed_tests > 0 else 'success'}">
                <h3>{summary.failed_tests}</h3>
                <p>失败测试</p>
            </div>
            <div class="card">
                <h3>{summary.success_rate:.1%}</h3>
                <p>成功率</p>
            </div>
        </div>
        
        {self._generate_performance_section(summary.performance_metrics)}
        
        {self._generate_coverage_section(summary.coverage_info)}
        
        <div class="section">
            <h2>📋 测试结果详情</h2>
            <div class="test-results">
                {self._generate_test_results_html(summary.test_results)}
            </div>
        </div>
        
        <div class="footer">
            <p>📊 报告由量化交易系统自动生成</p>
            <p>🔗 项目地址: <a href="#">quantification-agents</a></p>
        </div>
    </div>
</body>
</html>
        """
        
        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(output_file)
    
    def _generate_performance_section(self, metrics: Optional[PerformanceMetrics]) -> str:
        """生成性能指标部分"""
        if not metrics:
            return """
            <div class="section">
                <h2>⚡ 性能指标</h2>
                <p>暂无性能数据</p>
            </div>
            """
        
        target_results = metrics.meets_targets(self.performance_targets)
        
        metrics_html = ""
        
        if metrics.latency_avg_ms > 0:
            status = "pass" if target_results.get('latency', False) else "fail"
            metrics_html += f"""
                <div class="metric">
                    <div class="metric-name">平均延迟</div>
                    <div class="metric-value">
                        {metrics.latency_avg_ms:.2f}ms
                        <span class="metric-status status-{status}">
                            {"✓" if status == "pass" else "✗"} 目标 < {self.performance_targets['latency_target_ms']}ms
                        </span>
                    </div>
                </div>
            """
        
        if metrics.throughput_avg_tps > 0:
            status = "pass" if target_results.get('throughput', False) else "fail"
            metrics_html += f"""
                <div class="metric">
                    <div class="metric-name">平均吞吐量</div>
                    <div class="metric-value">
                        {metrics.throughput_avg_tps:.0f} TPS
                        <span class="metric-status status-{status}">
                            {"✓" if status == "pass" else "✗"} 目标 > {self.performance_targets['throughput_target_tps']}
                        </span>
                    </div>
                </div>
            """
        
        if metrics.memory_max_mb > 0:
            status = "pass" if target_results.get('memory', False) else "fail"
            metrics_html += f"""
                <div class="metric">
                    <div class="metric-name">最大内存使用</div>
                    <div class="metric-value">
                        {metrics.memory_max_mb:.1f}MB
                        <span class="metric-status status-{status}">
                            {"✓" if status == "pass" else "✗"} 目标 < {self.performance_targets['memory_target_mb']}MB
                        </span>
                    </div>
                </div>
            """
        
        return f"""
        <div class="section">
            <h2>⚡ 性能指标</h2>
            <div class="performance-grid">
                {metrics_html}
            </div>
        </div>
        """
    
    def _generate_coverage_section(self, coverage: Optional[CoverageInfo]) -> str:
        """生成覆盖率部分"""
        if not coverage:
            return """
            <div class="section">
                <h2>📈 代码覆盖率</h2>
                <p>暂无覆盖率数据</p>
            </div>
            """
        
        grade_class = f"grade-{coverage.coverage_grade.lower().replace('+', '')}"
        
        return f"""
        <div class="section">
            <h2>📈 代码覆盖率</h2>
            <div class="coverage-info">
                <div>
                    <strong>行覆盖率:</strong>
                    <span class="grade {grade_class}">{coverage.coverage_grade}</span>
                </div>
                <div class="coverage-bar">
                    <div class="coverage-fill" style="width: {coverage.line_coverage * 100}%"></div>
                </div>
                <div class="coverage-text">{coverage.line_coverage:.1%} ({coverage.covered_lines}/{coverage.total_lines} 行)</div>
                
                {f'<p><strong>分支覆盖率:</strong> {coverage.branch_coverage:.1%}</p>' if coverage.branch_coverage > 0 else ''}
            </div>
        </div>
        """
    
    def _generate_test_results_html(self, test_results: List[TestResult]) -> str:
        """生成测试结果HTML"""
        html = ""
        
        for result in test_results:
            status_class = {
                'passed': 'success',
                'failed': 'error',
                'error': 'error',
                'skipped': 'warning'
            }.get(result.status, 'warning')
            
            status_icon = {
                'passed': '✅',
                'failed': '❌',
                'error': '⚠️',
                'skipped': '⏸️'
            }.get(result.status, '❓')
            
            html += f"""
                <div class="test-result">
                    <div>
                        <div class="test-name">{status_icon} {result.test_name}</div>
                        <div class="test-class">{result.test_class}</div>
                    </div>
                    <div class="test-duration">{result.duration:.3f}s</div>
                </div>
            """
        
        return html
    
    def generate_json_summary(self, summary: TestSummary, output_file: str = None) -> str:
        """生成JSON格式的摘要报告"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.output_dir / f"test_summary_{timestamp}.json"
        
        # 转换为字典格式
        summary_dict = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': summary.total_tests,
            'passed_tests': summary.passed_tests,
            'failed_tests': summary.failed_tests,
            'skipped_tests': summary.skipped_tests,
            'error_tests': summary.error_tests,
            'success_rate': summary.success_rate,
            'failure_rate': summary.failure_rate,
            'total_duration': summary.total_duration,
            'performance_targets_met': {},
            'coverage_grade': None
        }
        
        # 添加性能指标
        if summary.performance_metrics:
            metrics = summary.performance_metrics
            summary_dict['performance_metrics'] = {
                'latency_avg_ms': metrics.latency_avg_ms,
                'latency_p95_ms': metrics.latency_p95_ms,
                'latency_p99_ms': metrics.latency_p99_ms,
                'throughput_avg_tps': metrics.throughput_avg_tps,
                'memory_max_mb': metrics.memory_max_mb,
                'cpu_max_percent': metrics.cpu_max_percent
            }
            summary_dict['performance_targets_met'] = metrics.meets_targets(self.performance_targets)
        
        # 添加覆盖率信息
        if summary.coverage_info:
            coverage = summary.coverage_info
            summary_dict['coverage_info'] = {
                'line_coverage': coverage.line_coverage,
                'branch_coverage': coverage.branch_coverage,
                'total_lines': coverage.total_lines,
                'covered_lines': coverage.covered_lines,
                'coverage_grade': coverage.coverage_grade
            }
            summary_dict['coverage_grade'] = coverage.coverage_grade
        
        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary_dict, f, indent=2, ensure_ascii=False)
        
        return str(output_file)


# 命令行工具
def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='生成策略管理系统测试报告')
    parser.add_argument('--json-report', help='pytest JSON报告文件路径')
    parser.add_argument('--coverage-xml', help='覆盖率XML报告文件路径')
    parser.add_argument('--output-dir', default='logs', help='输出目录')
    parser.add_argument('--format', choices=['html', 'json', 'both'], default='both', help='报告格式')
    
    args = parser.parse_args()
    
    generator = TestReportGenerator(args.output_dir)
    
    # 解析测试结果
    summary = TestSummary()
    
    if args.json_report and os.path.exists(args.json_report):
        summary = generator.parse_pytest_json_report(args.json_report)
        print(f"✅ 已解析pytest JSON报告: {args.json_report}")
    
    if args.coverage_xml and os.path.exists(args.coverage_xml):
        summary.coverage_info = generator.parse_coverage_xml_report(args.coverage_xml)
        print(f"✅ 已解析覆盖率XML报告: {args.coverage_xml}")
    
    # 提取性能指标
    if summary.test_results:
        summary.performance_metrics = generator.extract_performance_metrics(summary.test_results)
    
    # 生成报告
    output_files = []
    
    if args.format in ['html', 'both']:
        html_file = generator.generate_html_report(summary)
        output_files.append(html_file)
        print(f"✅ HTML报告已生成: {html_file}")
    
    if args.format in ['json', 'both']:
        json_file = generator.generate_json_summary(summary)
        output_files.append(json_file)
        print(f"✅ JSON摘要已生成: {json_file}")
    
    print(f"\n🎉 测试报告生成完成！")
    print(f"📊 测试摘要:")
    print(f"   总测试数: {summary.total_tests}")
    print(f"   成功率: {summary.success_rate:.1%}")
    if summary.coverage_info:
        print(f"   覆盖率: {summary.coverage_info.line_coverage:.1%} ({summary.coverage_info.coverage_grade})")


if __name__ == '__main__':
    main()