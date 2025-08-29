"""
Performance benchmarking and reporting framework for trajectory prediction models.

This module provides:
- Automated benchmark generation
- Performance report templates
- Model performance tracking over time
- Comparative analysis with baselines
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import logging
from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig

from ..models.base import TrajectoryPredictor
from ..data.schemas import TrajectoryData
from .evaluator import ModelEvaluator
from .comparison import ModelComparisonInterface

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Individual benchmark result for a model."""
    model_name: str
    timestamp: str
    dataset_info: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    computational_metrics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Comprehensive benchmark report."""
    report_id: str
    timestamp: str
    models: List[str]
    benchmark_results: List[BenchmarkResult]
    summary_statistics: Dict[str, Any]
    recommendations: List[str]
    report_path: Optional[str] = None


class PerformanceBenchmarker:
    """
    Automated performance benchmarking framework.
    
    Generates comprehensive performance reports and tracks model performance over time.
    """
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.output_dir = Path(config.get("benchmark_output_dir", "benchmark_reports"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize evaluator and comparison interface
        self.evaluator = ModelEvaluator(config)
        self.comparison_interface = ModelComparisonInterface(config)
        
        # Benchmark configuration
        self.benchmark_datasets = config.get("benchmark_datasets", [])
        self.standard_horizons = config.get("benchmark_horizons", [5.0, 10.0, 15.0])
        
        # Performance tracking
        self.history_file = self.output_dir / "benchmark_history.json"
        self.load_benchmark_history()
    
    def load_benchmark_history(self) -> None:
        """Load historical benchmark results."""
        self.benchmark_history = []
        
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    self.benchmark_history = json.load(f)
                logger.info(f"Loaded {len(self.benchmark_history)} historical benchmark records")
            except Exception as e:
                logger.warning(f"Failed to load benchmark history: {e}")
    
    def save_benchmark_history(self) -> None:
        """Save benchmark history to disk."""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.benchmark_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save benchmark history: {e}")
    
    async def run_comprehensive_benchmark(
        self,
        models: List[TrajectoryPredictor],
        test_datasets: Dict[str, List[TrajectoryData]],
        include_computational_metrics: bool = True,
        generate_report: bool = True
    ) -> BenchmarkReport:
        """
        Run comprehensive benchmark across multiple models and datasets.
        
        Args:
            models: List of models to benchmark
            test_datasets: Dictionary of dataset_name -> trajectory data
            include_computational_metrics: Whether to measure inference time, memory usage
            generate_report: Whether to generate detailed HTML/PDF report
            
        Returns:
            Comprehensive benchmark report
        """
        logger.info(f"Starting comprehensive benchmark of {len(models)} models across {len(test_datasets)} datasets")
        
        timestamp = datetime.now().isoformat()
        report_id = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        all_results = []
        
        # Benchmark each model on each dataset
        for model in models:
            logger.info(f"Benchmarking model: {model.name}")
            
            for dataset_name, trajectories in test_datasets.items():
                logger.info(f"  Dataset: {dataset_name} ({len(trajectories)} trajectories)")
                
                try:
                    benchmark_result = await self._benchmark_single_model(
                        model, trajectories, dataset_name, include_computational_metrics
                    )
                    all_results.append(benchmark_result)
                    
                except Exception as e:
                    logger.error(f"Benchmark failed for {model.name} on {dataset_name}: {e}")
                    
                    # Add failed result
                    failed_result = BenchmarkResult(
                        model_name=model.name,
                        timestamp=timestamp,
                        dataset_info={"name": dataset_name, "size": len(trajectories), "status": "failed"},
                        performance_metrics={"error": str(e)},
                        computational_metrics={},
                        metadata={"status": "failed"}
                    )
                    all_results.append(failed_result)
        
        # Generate summary statistics
        summary_stats = self._generate_summary_statistics(all_results)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_results, summary_stats)
        
        # Create benchmark report
        report = BenchmarkReport(
            report_id=report_id,
            timestamp=timestamp,
            models=[model.name for model in models],
            benchmark_results=all_results,
            summary_statistics=summary_stats,
            recommendations=recommendations
        )
        
        # Add to history
        self.benchmark_history.append({
            "report_id": report_id,
            "timestamp": timestamp,
            "models": report.models,
            "summary": summary_stats
        })
        self.save_benchmark_history()
        
        # Generate detailed report if requested
        if generate_report:
            report_path = await self._generate_detailed_report(report)
            report.report_path = str(report_path)
            logger.info(f"Detailed benchmark report generated: {report_path}")
        
        return report
    
    async def _benchmark_single_model(
        self,
        model: TrajectoryPredictor,
        trajectories: List[TrajectoryData],
        dataset_name: str,
        include_computational_metrics: bool = True
    ) -> BenchmarkResult:
        """Benchmark a single model on a dataset."""
        
        timestamp = datetime.now().isoformat()
        
        # Dataset information
        dataset_info = {
            "name": dataset_name,
            "size": len(trajectories),
            "avg_length": np.mean([len(t.positions) for t in trajectories]),
            "time_span": f"{min(t.time_steps[-1] for t in trajectories):.2f}-{max(t.time_steps[-1] for t in trajectories):.2f}s"
        }
        
        # Performance metrics across different horizons
        performance_metrics = {}
        computational_metrics = {}
        
        for horizon in self.standard_horizons:
            horizon_key = f"horizon_{horizon}s"
            
            # Performance evaluation
            try:
                eval_results = await self.evaluator.evaluate_model(
                    model, trajectories, horizon
                )
                performance_metrics[horizon_key] = eval_results
                
                # Computational metrics
                if include_computational_metrics:
                    comp_metrics = await self._measure_computational_performance(
                        model, trajectories, horizon
                    )
                    computational_metrics[horizon_key] = comp_metrics
                    
            except Exception as e:
                logger.warning(f"Failed to evaluate {model.name} at horizon {horizon}s: {e}")
                performance_metrics[horizon_key] = {"error": str(e)}
        
        return BenchmarkResult(
            model_name=model.name,
            timestamp=timestamp,
            dataset_info=dataset_info,
            performance_metrics=performance_metrics,
            computational_metrics=computational_metrics,
            metadata={"dataset": dataset_name, "status": "completed"}
        )
    
    async def _measure_computational_performance(
        self,
        model: TrajectoryPredictor,
        trajectories: List[TrajectoryData],
        prediction_horizon: float,
        n_samples: int = 100
    ) -> Dict[str, Any]:
        """Measure computational performance metrics."""
        
        import time
        import psutil
        import os
        
        # Sample trajectories for timing
        sample_trajectories = np.random.choice(trajectories, min(n_samples, len(trajectories)), replace=False)
        
        inference_times = []
        memory_usage = []
        
        process = psutil.Process(os.getpid())
        
        for trajectory in sample_trajectories:
            # Measure memory before
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Time prediction
            start_time = time.perf_counter()
            
            try:
                await model.predict_trajectory(trajectory, prediction_horizon)
                end_time = time.perf_counter()
                
                inference_time = (end_time - start_time) * 1000  # milliseconds
                inference_times.append(inference_time)
                
                # Measure memory after
                mem_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_usage.append(mem_after - mem_before)
                
            except Exception as e:
                logger.warning(f"Failed to measure performance for trajectory: {e}")
        
        if inference_times:
            return {
                "inference_time_ms": {
                    "mean": np.mean(inference_times),
                    "std": np.std(inference_times),
                    "min": np.min(inference_times),
                    "max": np.max(inference_times),
                    "p50": np.percentile(inference_times, 50),
                    "p95": np.percentile(inference_times, 95),
                    "p99": np.percentile(inference_times, 99)
                },
                "memory_usage_mb": {
                    "mean": np.mean(memory_usage),
                    "std": np.std(memory_usage),
                    "max": np.max(memory_usage)
                },
                "throughput_predictions_per_sec": 1000 / np.mean(inference_times) if np.mean(inference_times) > 0 else 0,
                "sample_size": len(inference_times)
            }
        else:
            return {"error": "No successful timing measurements"}
    
    def _generate_summary_statistics(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate summary statistics across all benchmark results."""
        
        summary = {
            "total_benchmarks": len(results),
            "successful_benchmarks": len([r for r in results if r.metadata.get("status") != "failed"]),
            "models": list(set(r.model_name for r in results)),
            "datasets": list(set(r.dataset_info["name"] for r in results if "name" in r.dataset_info)),
            "performance_summary": {},
            "computational_summary": {}
        }
        
        # Aggregate performance metrics
        successful_results = [r for r in results if r.metadata.get("status") != "failed"]
        
        if successful_results:
            # Performance metrics aggregation
            all_metrics = {}
            for result in successful_results:
                for horizon_key, metrics in result.performance_metrics.items():
                    if "error" not in metrics:
                        if horizon_key not in all_metrics:
                            all_metrics[horizon_key] = {}
                        
                        # Extract key metrics
                        for category in ["trajectory_metrics", "safety_metrics", "probabilistic_metrics"]:
                            if category in metrics:
                                for metric_name, metric_data in metrics[category].items():
                                    if isinstance(metric_data, dict) and "mean" in metric_data:
                                        metric_key = f"{category}_{metric_name}"
                                        if metric_key not in all_metrics[horizon_key]:
                                            all_metrics[horizon_key][metric_key] = []
                                        all_metrics[horizon_key][metric_key].append(metric_data["mean"])
            
            # Calculate statistics
            for horizon_key, horizon_metrics in all_metrics.items():
                summary["performance_summary"][horizon_key] = {}
                for metric_key, values in horizon_metrics.items():
                    if values:
                        summary["performance_summary"][horizon_key][metric_key] = {
                            "mean": np.mean(values),
                            "std": np.std(values),
                            "min": np.min(values),
                            "max": np.max(values),
                            "count": len(values)
                        }
            
            # Computational metrics aggregation
            comp_metrics = {}
            for result in successful_results:
                for horizon_key, metrics in result.computational_metrics.items():
                    if "error" not in metrics:
                        if horizon_key not in comp_metrics:
                            comp_metrics[horizon_key] = {}
                        
                        if "inference_time_ms" in metrics:
                            if "inference_time_ms" not in comp_metrics[horizon_key]:
                                comp_metrics[horizon_key]["inference_time_ms"] = []
                            comp_metrics[horizon_key]["inference_time_ms"].append(metrics["inference_time_ms"]["mean"])
                        
                        if "throughput_predictions_per_sec" in metrics:
                            if "throughput" not in comp_metrics[horizon_key]:
                                comp_metrics[horizon_key]["throughput"] = []
                            comp_metrics[horizon_key]["throughput"].append(metrics["throughput_predictions_per_sec"])
            
            # Calculate computational statistics
            for horizon_key, horizon_metrics in comp_metrics.items():
                summary["computational_summary"][horizon_key] = {}
                for metric_key, values in horizon_metrics.items():
                    if values:
                        summary["computational_summary"][horizon_key][metric_key] = {
                            "mean": np.mean(values),
                            "std": np.std(values),
                            "min": np.min(values),
                            "max": np.max(values)
                        }
        
        return summary
    
    def _generate_recommendations(
        self,
        results: List[BenchmarkResult],
        summary_stats: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable recommendations based on benchmark results."""
        
        recommendations = []
        
        # Check success rate
        success_rate = summary_stats["successful_benchmarks"] / summary_stats["total_benchmarks"]
        if success_rate < 0.9:
            recommendations.append(
                f"⚠️ Low benchmark success rate ({success_rate:.1%}). "
                "Review failed models for stability issues."
            )
        
        # Performance recommendations
        if "performance_summary" in summary_stats:
            # Find best performing models
            model_performance = {}
            for result in results:
                if result.metadata.get("status") != "failed":
                    model_name = result.model_name
                    if model_name not in model_performance:
                        model_performance[model_name] = []
                    
                    # Use ADE as primary metric
                    for horizon_metrics in result.performance_metrics.values():
                        if ("trajectory_metrics" in horizon_metrics and 
                            "ade" in horizon_metrics["trajectory_metrics"]):
                            ade_value = horizon_metrics["trajectory_metrics"]["ade"].get("mean")
                            if ade_value is not None:
                                model_performance[model_name].append(ade_value)
            
            # Calculate average ADE per model
            avg_performance = {}
            for model, ade_values in model_performance.items():
                if ade_values:
                    avg_performance[model] = np.mean(ade_values)
            
            if avg_performance:
                best_model = min(avg_performance, key=avg_performance.get)
                recommendations.append(
                    f"🏆 Best overall model: {best_model} "
                    f"(Average ADE: {avg_performance[best_model]:.3f})"
                )
                
                # Identify underperforming models
                worst_performance = max(avg_performance.values())
                best_performance = min(avg_performance.values())
                
                if worst_performance > best_performance * 2:  # Significant gap
                    underperforming = [
                        model for model, perf in avg_performance.items()
                        if perf > best_performance * 1.5
                    ]
                    if underperforming:
                        recommendations.append(
                            f"🔍 Consider reviewing models with high error rates: {', '.join(underperforming)}"
                        )
        
        # Computational recommendations
        if "computational_summary" in summary_stats:
            for horizon_key, metrics in summary_stats["computational_summary"].items():
                if "inference_time_ms" in metrics:
                    avg_inference_time = metrics["inference_time_ms"]["mean"]
                    
                    if avg_inference_time > 100:  # > 100ms is slow for real-time
                        recommendations.append(
                            f"⚡ High inference times detected ({avg_inference_time:.1f}ms avg at {horizon_key}). "
                            "Consider model optimization for real-time applications."
                        )
                    
                if "throughput" in metrics:
                    avg_throughput = metrics["throughput"]["mean"]
                    
                    if avg_throughput < 10:  # < 10 predictions/sec is low
                        recommendations.append(
                            f"📊 Low throughput detected ({avg_throughput:.1f} pred/sec at {horizon_key}). "
                            "Consider batching or model optimization."
                        )
        
        # Historical comparison
        if len(self.benchmark_history) > 1:
            recommendations.append(
                "📈 Historical performance tracking available. "
                "Consider trend analysis for performance degradation detection."
            )
        
        if not recommendations:
            recommendations.append("✅ All models performing within acceptable parameters.")
        
        return recommendations
    
    async def _generate_detailed_report(self, report: BenchmarkReport) -> Path:
        """Generate detailed HTML benchmark report."""
        
        report_dir = self.output_dir / f"report_{report.report_id}"
        report_dir.mkdir(exist_ok=True)
        
        # Generate visualizations
        await self._create_benchmark_visualizations(report, report_dir)
        
        # Create HTML report
        html_content = self._generate_html_report(report, report_dir)
        
        html_path = report_dir / "benchmark_report.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Save JSON data
        json_path = report_dir / "benchmark_data.json"
        with open(json_path, 'w') as f:
            json.dump({
                "report_id": report.report_id,
                "timestamp": report.timestamp,
                "models": report.models,
                "summary_statistics": report.summary_statistics,
                "recommendations": report.recommendations,
                "results": [
                    {
                        "model_name": r.model_name,
                        "timestamp": r.timestamp,
                        "dataset_info": r.dataset_info,
                        "performance_metrics": r.performance_metrics,
                        "computational_metrics": r.computational_metrics,
                        "metadata": r.metadata
                    }
                    for r in report.benchmark_results
                ]
            }, f, indent=2, default=str)
        
        return html_path
    
    async def _create_benchmark_visualizations(self, report: BenchmarkReport, output_dir: Path) -> None:
        """Create visualization charts for benchmark report."""
        
        # 1. Performance comparison chart
        await self._create_performance_comparison_chart(report, output_dir)
        
        # 2. Computational performance chart
        await self._create_computational_chart(report, output_dir)
        
        # 3. Historical trend chart (if available)
        if len(self.benchmark_history) > 1:
            await self._create_historical_trend_chart(output_dir)
    
    async def _create_performance_comparison_chart(self, report: BenchmarkReport, output_dir: Path) -> None:
        """Create performance comparison visualization."""
        
        # Extract data for plotting
        plot_data = []
        
        for result in report.benchmark_results:
            if result.metadata.get("status") != "failed":
                for horizon_key, metrics in result.performance_metrics.items():
                    if "error" not in metrics:
                        horizon = horizon_key.replace("horizon_", "").replace("s", "")
                        
                        # Extract key metrics
                        if "trajectory_metrics" in metrics:
                            for metric_name, metric_data in metrics["trajectory_metrics"].items():
                                if isinstance(metric_data, dict) and "mean" in metric_data:
                                    plot_data.append({
                                        "Model": result.model_name,
                                        "Dataset": result.dataset_info.get("name", "unknown"),
                                        "Horizon": f"{horizon}s",
                                        "Metric": metric_name.upper(),
                                        "Value": metric_data["mean"],
                                        "Category": "Trajectory"
                                    })
                        
                        if "safety_metrics" in metrics:
                            for metric_name, metric_data in metrics["safety_metrics"].items():
                                if isinstance(metric_data, dict) and "mean" in metric_data:
                                    plot_data.append({
                                        "Model": result.model_name,
                                        "Dataset": result.dataset_info.get("name", "unknown"),
                                        "Horizon": f"{horizon}s",
                                        "Metric": metric_name.upper(),
                                        "Value": metric_data["mean"],
                                        "Category": "Safety"
                                    })
        
        if not plot_data:
            return
        
        df = pd.DataFrame(plot_data)
        
        # Create subplots for key metrics
        key_metrics = ["ADE", "FDE", "COLLISION_RISK", "MIN_TTC"]
        available_metrics = [m for m in key_metrics if m in df["Metric"].values]
        
        if available_metrics:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()
            
            for idx, metric in enumerate(available_metrics[:4]):
                metric_data = df[df["Metric"] == metric]
                
                if len(metric_data) > 0 and idx < len(axes):
                    sns.barplot(data=metric_data, x="Model", y="Value", 
                              hue="Horizon", ax=axes[idx])
                    axes[idx].set_title(f"{metric} Performance Comparison")
                    axes[idx].tick_params(axis='x', rotation=45)
            
            # Hide empty subplots
            for idx in range(len(available_metrics), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(output_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    async def _create_computational_chart(self, report: BenchmarkReport, output_dir: Path) -> None:
        """Create computational performance visualization."""
        
        plot_data = []
        
        for result in report.benchmark_results:
            if result.metadata.get("status") != "failed":
                for horizon_key, metrics in result.computational_metrics.items():
                    if "error" not in metrics:
                        horizon = horizon_key.replace("horizon_", "").replace("s", "")
                        
                        if "inference_time_ms" in metrics:
                            plot_data.append({
                                "Model": result.model_name,
                                "Horizon": f"{horizon}s",
                                "Inference Time (ms)": metrics["inference_time_ms"]["mean"]
                            })
        
        if not plot_data:
            return
        
        df = pd.DataFrame(plot_data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df, x="Model", y="Inference Time (ms)", hue="Horizon", ax=ax)
        ax.set_title("Computational Performance Comparison")
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / "computational_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    async def _create_historical_trend_chart(self, output_dir: Path) -> None:
        """Create historical performance trend chart."""
        
        if len(self.benchmark_history) < 2:
            return
        
        # Extract historical data
        historical_data = []
        
        for record in self.benchmark_history[-10:]:  # Last 10 records
            timestamp = pd.to_datetime(record["timestamp"])
            
            if "summary" in record and "performance_summary" in record["summary"]:
                for model in record["models"]:
                    # Try to extract ADE from summary
                    for horizon_key, metrics in record["summary"]["performance_summary"].items():
                        if "trajectory_metrics_ade" in metrics:
                            historical_data.append({
                                "Timestamp": timestamp,
                                "Model": model,
                                "Horizon": horizon_key,
                                "ADE": metrics["trajectory_metrics_ade"]["mean"]
                            })
        
        if historical_data:
            df = pd.DataFrame(historical_data)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for model in df["Model"].unique():
                model_data = df[df["Model"] == model]
                ax.plot(model_data["Timestamp"], model_data["ADE"], 
                       marker='o', label=model, linewidth=2)
            
            ax.set_xlabel("Date")
            ax.set_ylabel("Average Displacement Error (ADE)")
            ax.set_title("Historical Performance Trends")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / "historical_trends.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_html_report(self, report: BenchmarkReport, report_dir: Path) -> str:
        """Generate HTML report content."""
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trajectory Prediction Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .summary-box {{ background: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .metric-card {{ background: white; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; }}
        .recommendations {{ background: #e8f6f3; border-left: 4px solid #1abc9c; padding: 20px; margin: 20px 0; }}
        .recommendations ul {{ margin: 0; }}
        .chart {{ text-align: center; margin: 20px 0; }}
        .chart img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .status-success {{ color: #27ae60; font-weight: bold; }}
        .status-failed {{ color: #e74c3c; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚗 Trajectory Prediction Benchmark Report</h1>
        
        <div class="summary-box">
            <h2>📊 Executive Summary</h2>
            <p><strong>Report ID:</strong> {report.report_id}</p>
            <p><strong>Generated:</strong> {pd.to_datetime(report.timestamp).strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Models Tested:</strong> {len(report.models)}</p>
            <p><strong>Total Benchmarks:</strong> {report.summary_statistics.get('total_benchmarks', 'N/A')}</p>
            <p><strong>Success Rate:</strong> {(report.summary_statistics.get('successful_benchmarks', 0) / report.summary_statistics.get('total_benchmarks', 1) * 100):.1f}%</p>
        </div>
        
        <h2>🏆 Key Recommendations</h2>
        <div class="recommendations">
            <ul>
                {"".join(f"<li>{rec}</li>" for rec in report.recommendations)}
            </ul>
        </div>
        
        <h2>📈 Performance Visualizations</h2>
        
        <div class="chart">
            <h3>Model Performance Comparison</h3>
            <img src="performance_comparison.png" alt="Performance Comparison" />
        </div>
        
        <div class="chart">
            <h3>Computational Performance</h3>
            <img src="computational_performance.png" alt="Computational Performance" />
        </div>
        
        {"<div class='chart'><h3>Historical Performance Trends</h3><img src='historical_trends.png' alt='Historical Trends' /></div>" if (report_dir / "historical_trends.png").exists() else ""}
        
        <h2>📋 Detailed Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Dataset</th>
                    <th>Status</th>
                    <th>ADE (10s)</th>
                    <th>FDE (10s)</th>
                    <th>Collision Risk</th>
                    <th>Avg Inference (ms)</th>
                </tr>
            </thead>
            <tbody>
                {self._generate_results_table_rows(report.benchmark_results)}
            </tbody>
        </table>
        
        <h2>🔧 Technical Details</h2>
        <p>This benchmark report was generated using the Trajectory Prediction Evaluation Framework. 
        All metrics are computed using standard evaluation procedures with cross-validation.</p>
        
        <p><small>Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small></p>
    </div>
</body>
</html>
        """
        
        return html_template
    
    def _generate_results_table_rows(self, results: List[BenchmarkResult]) -> str:
        """Generate HTML table rows for benchmark results."""
        
        rows = []
        
        for result in results:
            status = "✅ Success" if result.metadata.get("status") != "failed" else "❌ Failed"
            status_class = "status-success" if result.metadata.get("status") != "failed" else "status-failed"
            
            # Extract metrics
            ade_10s = "N/A"
            fde_10s = "N/A"
            collision_risk = "N/A"
            inference_time = "N/A"
            
            horizon_10s = result.performance_metrics.get("horizon_10.0s", {})
            if "trajectory_metrics" in horizon_10s:
                if "ade" in horizon_10s["trajectory_metrics"]:
                    ade_data = horizon_10s["trajectory_metrics"]["ade"]
                    if isinstance(ade_data, dict) and "mean" in ade_data:
                        ade_10s = f"{ade_data['mean']:.3f}"
                
                if "fde" in horizon_10s["trajectory_metrics"]:
                    fde_data = horizon_10s["trajectory_metrics"]["fde"]
                    if isinstance(fde_data, dict) and "mean" in fde_data:
                        fde_10s = f"{fde_data['mean']:.3f}"
            
            if "safety_metrics" in horizon_10s:
                if "collision_risk" in horizon_10s["safety_metrics"]:
                    cr_data = horizon_10s["safety_metrics"]["collision_risk"]
                    if isinstance(cr_data, dict) and "mean" in cr_data:
                        collision_risk = f"{cr_data['mean']:.3f}"
            
            comp_10s = result.computational_metrics.get("horizon_10.0s", {})
            if "inference_time_ms" in comp_10s:
                it_data = comp_10s["inference_time_ms"]
                if isinstance(it_data, dict) and "mean" in it_data:
                    inference_time = f"{it_data['mean']:.1f}"
            
            row = f"""
                <tr>
                    <td>{result.model_name}</td>
                    <td>{result.dataset_info.get('name', 'Unknown')}</td>
                    <td class="{status_class}">{status}</td>
                    <td>{ade_10s}</td>
                    <td>{fde_10s}</td>
                    <td>{collision_risk}</td>
                    <td>{inference_time}</td>
                </tr>
            """
            rows.append(row)
        
        return "".join(rows)