"""
集成测试报告生成器
生成详细的HTML和JSON格式测试报告
"""

import json
import os
import statistics
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import xml.etree.ElementTree as ET

import pytest


class TestReportGenerator:
    """测试报告生成器"""
    
    def __init__(self, report_dir: str = None):
        self.report_dir = Path(report_dir) if report_dir else Path("test_reports")
        self.report_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.test_results = []
        self.performance_metrics = {}
        self.security_results = {}
        self.coverage_data = {}
    
    def add_test_result(self, 
                       test_name: str, 
                       status: str, 
                       duration: float, 
                       details: Dict[str, Any] = None):
        """添加测试结果"""
        result = {
            'test_name': test_name,
            'status': status,  # 'passed', 'failed', 'error', 'skipped'
            'duration': duration,
            'timestamp': datetime.now().isoformat(),
            'details': details or {}
        }
        self.test_results.append(result)
    
    def add_performance_metrics(self, 
                              test_name: str, 
                              metrics: Dict[str, Any]):
        """添加性能指标"""
        self.performance_metrics[test_name] = {
            **metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    def add_security_results(self, 
                           test_name: str, 
                           results: Dict[str, Any]):
        """添加安全测试结果"""
        self.security_results[test_name] = {
            **results,
            'timestamp': datetime.now().isoformat()
        }
    
    def parse_junit_xml(self, junit_file: str) -> Dict[str, Any]:
        """解析JUnit XML报告"""
        if not os.path.exists(junit_file):
            return {}
        
        try:
            tree = ET.parse(junit_file)
            root = tree.getroot()
            
            # 提取测试套件信息
            testsuite = root if root.tag == 'testsuite' else root.find('testsuite')
            if testsuite is None:
                return {}
            
            suite_data = {
                'name': testsuite.get('name', ''),
                'tests': int(testsuite.get('tests', 0)),
                'failures': int(testsuite.get('failures', 0)),
                'errors': int(testsuite.get('errors', 0)),
                'skipped': int(testsuite.get('skipped', 0)),
                'time': float(testsuite.get('time', 0)),
                'testcases': []
            }
            
            # 提取测试用例信息
            for testcase in testsuite.findall('testcase'):
                case_data = {
                    'name': testcase.get('name', ''),
                    'classname': testcase.get('classname', ''),
                    'time': float(testcase.get('time', 0)),
                    'status': 'passed'
                }
                
                # 检查失败、错误或跳过
                if testcase.find('failure') is not None:
                    case_data['status'] = 'failed'
                    case_data['failure'] = testcase.find('failure').text
                elif testcase.find('error') is not None:
                    case_data['status'] = 'error'
                    case_data['error'] = testcase.find('error').text
                elif testcase.find('skipped') is not None:
                    case_data['status'] = 'skipped'
                    case_data['skip_reason'] = testcase.find('skipped').text
                
                suite_data['testcases'].append(case_data)
            
            return suite_data
            
        except ET.ParseError as e:
            print(f"解析JUnit XML文件失败: {e}")
            return {}
    
    def parse_coverage_xml(self, coverage_file: str) -> Dict[str, Any]:
        """解析覆盖率XML报告"""
        if not os.path.exists(coverage_file):
            return {}
        
        try:
            tree = ET.parse(coverage_file)
            root = tree.getroot()
            
            coverage_data = {
                'line_rate': float(root.get('line-rate', 0)),
                'branch_rate': float(root.get('branch-rate', 0)),
                'lines_covered': int(root.get('lines-covered', 0)),
                'lines_valid': int(root.get('lines-valid', 0)),
                'branches_covered': int(root.get('branches-covered', 0)),
                'branches_valid': int(root.get('branches-valid', 0)),
                'complexity': float(root.get('complexity', 0)),
                'timestamp': root.get('timestamp', ''),
                'packages': []
            }
            
            # 提取包级别的覆盖率信息
            packages = root.find('packages')
            if packages is not None:
                for package in packages.findall('package'):
                    package_data = {
                        'name': package.get('name', ''),
                        'line_rate': float(package.get('line-rate', 0)),
                        'branch_rate': float(package.get('branch-rate', 0)),
                        'complexity': float(package.get('complexity', 0)),
                        'classes': []
                    }
                    
                    classes = package.find('classes')
                    if classes is not None:
                        for cls in classes.findall('class'):
                            class_data = {
                                'name': cls.get('name', ''),
                                'filename': cls.get('filename', ''),
                                'line_rate': float(cls.get('line-rate', 0)),
                                'branch_rate': float(cls.get('branch-rate', 0)),
                                'complexity': float(cls.get('complexity', 0))
                            }
                            package_data['classes'].append(class_data)
                    
                    coverage_data['packages'].append(package_data)
            
            return coverage_data
            
        except ET.ParseError as e:
            print(f"解析覆盖率XML文件失败: {e}")
            return {}
    
    def calculate_summary_stats(self) -> Dict[str, Any]:
        """计算汇总统计信息"""
        if not self.test_results:
            return {}
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r['status'] == 'passed'])
        failed_tests = len([r for r in self.test_results if r['status'] == 'failed'])
        error_tests = len([r for r in self.test_results if r['status'] == 'error'])
        skipped_tests = len([r for r in self.test_results if r['status'] == 'skipped'])
        
        durations = [r['duration'] for r in self.test_results if r['duration'] > 0]
        
        stats = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'error_tests': error_tests,
            'skipped_tests': skipped_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'total_duration': sum(durations),
            'avg_duration': statistics.mean(durations) if durations else 0,
            'median_duration': statistics.median(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
            'min_duration': min(durations) if durations else 0
        }
        
        return stats
    
    def generate_json_report(self) -> str:
        """生成JSON格式报告"""
        report_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'timestamp': self.timestamp,
                'report_type': 'integration_tests',
                'version': '1.0'
            },
            'summary': self.calculate_summary_stats(),
            'test_results': self.test_results,
            'performance_metrics': self.performance_metrics,
            'security_results': self.security_results,
            'coverage_data': self.coverage_data
        }
        
        json_file = self.report_dir / f"integration_report_{self.timestamp}.json"
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        return str(json_file)
    
    def generate_html_report(self) -> str:
        """生成HTML格式报告"""
        summary = self.calculate_summary_stats()
        
        html_file = self.report_dir / f"integration_report_{self.timestamp}.html"
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(self._generate_html_content(summary))
        
        return str(html_file)
    
    def _generate_html_content(self, summary: Dict[str, Any]) -> str:
        """生成HTML内容"""
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>REST API & WebSocket 集成测试报告</title>
    <style>
        {self._get_css_styles()}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        {self._generate_header()}
        {self._generate_summary_section(summary)}
        {self._generate_test_results_section()}
        {self._generate_performance_section()}
        {self._generate_security_section()}
        {self._generate_coverage_section()}
        {self._generate_recommendations_section()}
        {self._generate_footer()}
    </div>
    
    <script>
        {self._generate_charts_script()}
    </script>
</body>
</html>"""
    
    def _get_css_styles(self) -> str:
        """获取CSS样式"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .section {
            background: white;
            margin-bottom: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .section-header {
            background: #f8f9fa;
            padding: 20px 30px;
            border-bottom: 1px solid #dee2e6;
        }
        
        .section-header h2 {
            color: #495057;
            font-size: 1.5em;
        }
        
        .section-content {
            padding: 30px;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .metric-card {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #007bff;
        }
        
        .metric-card.success {
            border-left-color: #28a745;
        }
        
        .metric-card.warning {
            border-left-color: #ffc107;
        }
        
        .metric-card.danger {
            border-left-color: #dc3545;
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #495057;
        }
        
        .metric-label {
            color: #6c757d;
            font-size: 0.9em;
            margin-top: 5px;
        }
        
        .test-list {
            list-style: none;
        }
        
        .test-item {
            padding: 15px;
            border-bottom: 1px solid #dee2e6;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .test-item:last-child {
            border-bottom: none;
        }
        
        .test-name {
            font-weight: 500;
        }
        
        .test-status {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 500;
        }
        
        .status-passed {
            background: #d4edda;
            color: #155724;
        }
        
        .status-failed {
            background: #f8d7da;
            color: #721c24;
        }
        
        .status-error {
            background: #f8d7da;
            color: #721c24;
        }
        
        .status-skipped {
            background: #fff3cd;
            color: #856404;
        }
        
        .chart-container {
            position: relative;
            height: 300px;
            margin: 20px 0;
        }
        
        .performance-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        .performance-table th,
        .performance-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }
        
        .performance-table th {
            background: #f8f9fa;
            font-weight: 600;
        }
        
        .recommendations {
            background: #e3f2fd;
            border: 1px solid #bbdefb;
            border-radius: 8px;
            padding: 20px;
        }
        
        .recommendations h3 {
            color: #0d47a1;
            margin-bottom: 15px;
        }
        
        .recommendations ul {
            list-style-position: inside;
            color: #1565c0;
        }
        
        .recommendations li {
            margin-bottom: 8px;
        }
        
        .footer {
            text-align: center;
            padding: 30px;
            color: #6c757d;
            border-top: 1px solid #dee2e6;
            margin-top: 50px;
        }
        """
    
    def _generate_header(self) -> str:
        """生成报告头部"""
        return f"""
        <div class="header">
            <h1>REST API & WebSocket 集成测试报告</h1>
            <p>生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
            <p>报告版本: {self.timestamp}</p>
        </div>
        """
    
    def _generate_summary_section(self, summary: Dict[str, Any]) -> str:
        """生成汇总部分"""
        if not summary:
            return ""
        
        success_rate = summary.get('success_rate', 0) * 100
        total_duration = summary.get('total_duration', 0)
        
        return f"""
        <div class="section">
            <div class="section-header">
                <h2>测试执行汇总</h2>
            </div>
            <div class="section-content">
                <div class="metrics-grid">
                    <div class="metric-card success">
                        <div class="metric-value">{summary.get('total_tests', 0)}</div>
                        <div class="metric-label">总测试数</div>
                    </div>
                    <div class="metric-card success">
                        <div class="metric-value">{summary.get('passed_tests', 0)}</div>
                        <div class="metric-label">通过测试</div>
                    </div>
                    <div class="metric-card danger">
                        <div class="metric-value">{summary.get('failed_tests', 0)}</div>
                        <div class="metric-label">失败测试</div>
                    </div>
                    <div class="metric-card warning">
                        <div class="metric-value">{summary.get('skipped_tests', 0)}</div>
                        <div class="metric-label">跳过测试</div>
                    </div>
                    <div class="metric-card success">
                        <div class="metric-value">{success_rate:.1f}%</div>
                        <div class="metric-label">成功率</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{total_duration:.2f}s</div>
                        <div class="metric-label">总耗时</div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <canvas id="summaryChart"></canvas>
                </div>
            </div>
        </div>
        """
    
    def _generate_test_results_section(self) -> str:
        """生成测试结果部分"""
        if not self.test_results:
            return ""
        
        test_items = []
        for result in self.test_results:
            status_class = f"status-{result['status']}"
            status_text = {
                'passed': '通过',
                'failed': '失败', 
                'error': '错误',
                'skipped': '跳过'
            }.get(result['status'], result['status'])
            
            test_items.append(f"""
                <li class="test-item">
                    <div>
                        <div class="test-name">{result['test_name']}</div>
                        <div class="test-duration">{result['duration']:.3f}s</div>
                    </div>
                    <div class="test-status {status_class}">{status_text}</div>
                </li>
            """)
        
        return f"""
        <div class="section">
            <div class="section-header">
                <h2>详细测试结果</h2>
            </div>
            <div class="section-content">
                <ul class="test-list">
                    {''.join(test_items)}
                </ul>
            </div>
        </div>
        """
    
    def _generate_performance_section(self) -> str:
        """生成性能测试部分"""
        if not self.performance_metrics:
            return ""
        
        table_rows = []
        for test_name, metrics in self.performance_metrics.items():
            avg_time = metrics.get('avg_response_time', 0) * 1000  # 转换为毫秒
            p95_time = metrics.get('p95_response_time', 0) * 1000
            throughput = metrics.get('throughput_rps', 0)
            
            table_rows.append(f"""
                <tr>
                    <td>{test_name}</td>
                    <td>{avg_time:.2f}ms</td>
                    <td>{p95_time:.2f}ms</td>
                    <td>{throughput:.2f} RPS</td>
                    <td>{metrics.get('success_rate', 0):.1%}</td>
                </tr>
            """)
        
        return f"""
        <div class="section">
            <div class="section-header">
                <h2>性能测试结果</h2>
            </div>
            <div class="section-content">
                <table class="performance-table">
                    <thead>
                        <tr>
                            <th>测试名称</th>
                            <th>平均响应时间</th>
                            <th>P95响应时间</th>
                            <th>吞吐量</th>
                            <th>成功率</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(table_rows)}
                    </tbody>
                </table>
                
                <div class="chart-container">
                    <canvas id="performanceChart"></canvas>
                </div>
            </div>
        </div>
        """
    
    def _generate_security_section(self) -> str:
        """生成安全测试部分"""
        if not self.security_results:
            return ""
        
        security_items = []
        for test_name, results in self.security_results.items():
            block_rate = results.get('block_rate', 0) * 100
            total_attacks = results.get('total_attacks', 0)
            blocked_attacks = results.get('blocked_attacks', 0)
            
            security_items.append(f"""
                <div class="metric-card">
                    <h4>{test_name}</h4>
                    <p>攻击总数: {total_attacks}</p>
                    <p>阻止数量: {blocked_attacks}</p>
                    <p>阻止率: {block_rate:.1f}%</p>
                </div>
            """)
        
        return f"""
        <div class="section">
            <div class="section-header">
                <h2>安全测试结果</h2>
            </div>
            <div class="section-content">
                <div class="metrics-grid">
                    {''.join(security_items)}
                </div>
            </div>
        </div>
        """
    
    def _generate_coverage_section(self) -> str:
        """生成覆盖率部分"""
        if not self.coverage_data:
            return ""
        
        line_rate = self.coverage_data.get('line_rate', 0) * 100
        branch_rate = self.coverage_data.get('branch_rate', 0) * 100
        
        return f"""
        <div class="section">
            <div class="section-header">
                <h2>代码覆盖率</h2>
            </div>
            <div class="section-content">
                <div class="metrics-grid">
                    <div class="metric-card success">
                        <div class="metric-value">{line_rate:.1f}%</div>
                        <div class="metric-label">行覆盖率</div>
                    </div>
                    <div class="metric-card success">
                        <div class="metric-value">{branch_rate:.1f}%</div>
                        <div class="metric-label">分支覆盖率</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{self.coverage_data.get('lines_covered', 0)}</div>
                        <div class="metric-label">覆盖行数</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{self.coverage_data.get('lines_valid', 0)}</div>
                        <div class="metric-label">有效行数</div>
                    </div>
                </div>
            </div>
        </div>
        """
    
    def _generate_recommendations_section(self) -> str:
        """生成建议部分"""
        summary = self.calculate_summary_stats()
        recommendations = []
        
        if summary:
            success_rate = summary.get('success_rate', 0)
            if success_rate < 0.9:
                recommendations.append("测试成功率低于90%，需要检查失败的测试用例并修复相关问题")
            
            avg_duration = summary.get('avg_duration', 0)
            if avg_duration > 1.0:
                recommendations.append("平均测试执行时间过长，考虑优化测试性能或并行执行")
        
        # 性能建议
        for test_name, metrics in self.performance_metrics.items():
            avg_time = metrics.get('avg_response_time', 0)
            if 'api' in test_name.lower() and avg_time > 0.1:
                recommendations.append(f"{test_name}的API响应时间超过100ms，需要优化")
            elif 'websocket' in test_name.lower() and avg_time > 0.05:
                recommendations.append(f"{test_name}的WebSocket延迟超过50ms，需要优化")
        
        # 安全建议
        for test_name, results in self.security_results.items():
            block_rate = results.get('block_rate', 0)
            if block_rate < 0.8:
                recommendations.append(f"{test_name}的安全防护阻止率低于80%，需要加强安全措施")
        
        # 覆盖率建议
        if self.coverage_data:
            line_rate = self.coverage_data.get('line_rate', 0)
            if line_rate < 0.85:
                recommendations.append("代码覆盖率低于85%，需要增加更多测试用例")
        
        if not recommendations:
            recommendations.append("所有指标都在正常范围内，继续保持良好的测试实践")
        
        rec_items = [f"<li>{rec}</li>" for rec in recommendations]
        
        return f"""
        <div class="section">
            <div class="section-header">
                <h2>优化建议</h2>
            </div>
            <div class="section-content">
                <div class="recommendations">
                    <h3>根据测试结果，我们建议：</h3>
                    <ul>
                        {''.join(rec_items)}
                    </ul>
                </div>
            </div>
        </div>
        """
    
    def _generate_footer(self) -> str:
        """生成页脚"""
        return f"""
        <div class="footer">
            <p>© {datetime.now().year} 量化交易系统 - 集成测试报告</p>
            <p>报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """
    
    def _generate_charts_script(self) -> str:
        """生成图表脚本"""
        summary = self.calculate_summary_stats()
        
        if not summary:
            return ""
        
        return f"""
        // 测试结果饼图
        const summaryCtx = document.getElementById('summaryChart');
        if (summaryCtx) {{
            new Chart(summaryCtx, {{
                type: 'pie',
                data: {{
                    labels: ['通过', '失败', '错误', '跳过'],
                    datasets: [{{
                        data: [
                            {summary.get('passed_tests', 0)},
                            {summary.get('failed_tests', 0)},
                            {summary.get('error_tests', 0)},
                            {summary.get('skipped_tests', 0)}
                        ],
                        backgroundColor: [
                            '#28a745',
                            '#dc3545',
                            '#fd7e14',
                            '#ffc107'
                        ]
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        title: {{
                            display: true,
                            text: '测试结果分布'
                        }}
                    }}
                }}
            }});
        }}
        
        // 性能图表
        const performanceCtx = document.getElementById('performanceChart');
        if (performanceCtx) {{
            const perfData = {json.dumps(self.performance_metrics)};
            const labels = Object.keys(perfData);
            const responseTimeData = labels.map(label => 
                (perfData[label].avg_response_time || 0) * 1000
            );
            
            new Chart(performanceCtx, {{
                type: 'bar',
                data: {{
                    labels: labels,
                    datasets: [{{
                        label: '平均响应时间 (ms)',
                        data: responseTimeData,
                        backgroundColor: '#007bff'
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        title: {{
                            display: true,
                            text: '性能测试结果'
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true
                        }}
                    }}
                }}
            }});
        }}
        """
    
    def generate_full_report(self, 
                           junit_files: List[str] = None,
                           coverage_file: str = None) -> Dict[str, str]:
        """生成完整报告"""
        # 解析JUnit文件
        if junit_files:
            for junit_file in junit_files:
                suite_data = self.parse_junit_xml(junit_file)
                if suite_data:
                    # 添加测试结果
                    for testcase in suite_data.get('testcases', []):
                        self.add_test_result(
                            test_name=testcase['name'],
                            status=testcase['status'],
                            duration=testcase['time'],
                            details={'classname': testcase['classname']}
                        )
        
        # 解析覆盖率文件
        if coverage_file:
            self.coverage_data = self.parse_coverage_xml(coverage_file)
        
        # 生成报告文件
        json_report = self.generate_json_report()
        html_report = self.generate_html_report()
        
        return {
            'json_report': json_report,
            'html_report': html_report,
            'timestamp': self.timestamp
        }


# 示例用法
if __name__ == "__main__":
    # 创建报告生成器
    generator = TestReportGenerator("test_reports")
    
    # 添加示例数据
    generator.add_test_result("test_api_authentication", "passed", 0.05)
    generator.add_test_result("test_websocket_connection", "passed", 0.02)
    generator.add_test_result("test_rate_limiting", "failed", 0.15)
    
    generator.add_performance_metrics("API性能测试", {
        'avg_response_time': 0.08,
        'p95_response_time': 0.12,
        'throughput_rps': 150.5,
        'success_rate': 0.98
    })
    
    generator.add_security_results("SQL注入防护测试", {
        'total_attacks': 100,
        'blocked_attacks': 95,
        'block_rate': 0.95
    })
    
    # 生成报告
    reports = generator.generate_full_report()
    
    print("报告已生成:")
    for report_type, report_path in reports.items():
        print(f"  {report_type}: {report_path}")