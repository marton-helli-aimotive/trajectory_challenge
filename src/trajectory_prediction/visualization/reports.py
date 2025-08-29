"""
Automated reporting system for trajectory prediction analysis.

This module provides:
- AutomatedReportGenerator: Generate comprehensive analysis reports
- DataQualityReporter: Data quality assessment reports
- ModelComparisonReporter: Model performance comparison reports
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
import base64
from io import BytesIO
import warnings

from ..data.schemas import TrajectoryData
from ..evaluation.metrics import TrajectoryMetrics, SafetyMetrics, ProbabilisticMetrics
from ..api.models import TrajectoryResponse
from .components import PlotConfig
from .analysis import TrajectoryClusterAnalyzer, AnomalyDetector


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    
    include_plots: bool = True
    plot_format: str = "html"  # html, png, jpg, svg
    include_raw_data: bool = False
    max_plots_per_section: int = 5
    export_format: str = "html"  # html, pdf, json
    title: str = "Trajectory Prediction Analysis Report"
    author: str = "Trajectory Prediction System"
    logo_path: Optional[str] = None


class AutomatedReportGenerator:
    """Generate comprehensive analysis reports automatically."""
    
    def __init__(
        self,
        config: Optional[ReportConfig] = None,
        plot_config: Optional[PlotConfig] = None
    ):
        self.config = config or ReportConfig()
        self.plot_config = plot_config or PlotConfig()
        
    def generate_comprehensive_report(
        self,
        trajectories: List[TrajectoryData],
        predictions: Dict[str, List[TrajectoryResponse]],
        evaluation_results: Dict[str, Dict[str, float]],
        clustering_results: Optional[Dict] = None,
        anomaly_results: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report.
        
        Args:
            trajectories: Input trajectory data
            predictions: Model predictions by model name
            evaluation_results: Evaluation metrics by model
            clustering_results: Clustering analysis results
            anomaly_results: Anomaly detection results
            
        Returns:
            Dictionary containing complete report data
        """
        report = {
            'metadata': self._generate_metadata(),
            'executive_summary': self._generate_executive_summary(
                trajectories, predictions, evaluation_results
            ),
            'data_overview': self._generate_data_overview(trajectories),
            'model_performance': self._generate_model_performance_section(
                predictions, evaluation_results
            ),
            'detailed_analysis': self._generate_detailed_analysis(
                trajectories, predictions, clustering_results, anomaly_results
            ),
            'recommendations': self._generate_recommendations(
                evaluation_results, clustering_results, anomaly_results
            ),
            'appendix': self._generate_appendix(trajectories, predictions)
        }
        
        return report
    
    def export_report(
        self,
        report: Dict[str, Any],
        output_path: str,
        format: str = None
    ) -> str:
        """Export report to specified format."""
        format = format or self.config.export_format
        
        if format == "html":
            return self._export_html_report(report, output_path)
        elif format == "json":
            return self._export_json_report(report, output_path)
        elif format == "pdf":
            return self._export_pdf_report(report, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _generate_metadata(self) -> Dict[str, Any]:
        """Generate report metadata."""
        return {
            'title': self.config.title,
            'author': self.config.author,
            'generated_at': datetime.now().isoformat(),
            'version': "1.0.0",
            'report_type': "comprehensive_analysis"
        }
    
    def _generate_executive_summary(
        self,
        trajectories: List[TrajectoryData],
        predictions: Dict[str, List[TrajectoryResponse]],
        evaluation_results: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """Generate executive summary section."""
        n_trajectories = len(trajectories)
        n_models = len(predictions)
        
        # Find best performing model
        best_model = None
        best_score = float('inf')
        
        for model_name, metrics in evaluation_results.items():
            rmse = metrics.get('rmse', float('inf'))
            if rmse < best_score:
                best_score = rmse
                best_model = model_name
        
        # Calculate average metrics across models
        avg_metrics = {}
        for metric_name in ['rmse', 'mae', 'ade', 'fde']:
            values = [metrics.get(metric_name, 0) for metrics in evaluation_results.values()]
            avg_metrics[metric_name] = np.mean(values) if values else 0
        
        summary = {
            'key_findings': [
                f"Analyzed {n_trajectories:,} trajectories across {n_models} models",
                f"Best performing model: {best_model} (RMSE: {best_score:.3f})" if best_model else "No valid model results",
                f"Average RMSE across models: {avg_metrics['rmse']:.3f}",
                f"Average prediction horizon: {self._calculate_avg_prediction_horizon(predictions):.1f}s"
            ],
            'performance_overview': avg_metrics,
            'data_quality_score': self._calculate_data_quality_score(trajectories),
            'recommendations_summary': [
                "Deploy best performing model for production use" if best_model else "No model ready for deployment",
                "Monitor prediction accuracy continuously",
                "Consider ensemble methods for improved robustness"
            ]
        }
        
        return summary
    
    def _generate_data_overview(self, trajectories: List[TrajectoryData]) -> Dict[str, Any]:
        """Generate data overview section."""
        if not trajectories:
            return {'error': 'No trajectory data available'}
        
        # Basic statistics
        lengths = [len(traj.positions) for traj in trajectories]
        durations = []
        speeds = []
        
        for traj in trajectories:
            if len(traj.timestamps) > 1:
                durations.append(traj.timestamps[-1] - traj.timestamps[0])
            
            if traj.velocities:
                speeds.extend([v.magnitude for v in traj.velocities])
        
        overview = {
            'dataset_statistics': {
                'total_trajectories': len(trajectories),
                'avg_trajectory_length': np.mean(lengths),
                'avg_duration': np.mean(durations) if durations else 0,
                'avg_speed': np.mean(speeds) if speeds else 0,
                'max_speed': np.max(speeds) if speeds else 0,
                'trajectory_length_std': np.std(lengths)
            },
            'data_coverage': {
                'spatial_range': self._calculate_spatial_range(trajectories),
                'temporal_range': self._calculate_temporal_range(trajectories),
                'speed_distribution': self._calculate_speed_distribution(speeds) if speeds else {}
            },
            'quality_indicators': {
                'complete_trajectories': sum(1 for traj in trajectories if self._is_complete_trajectory(traj)),
                'missing_data_rate': self._calculate_missing_data_rate(trajectories),
                'outlier_rate': self._calculate_outlier_rate(trajectories)
            }
        }
        
        return overview
    
    def _generate_model_performance_section(
        self,
        predictions: Dict[str, List[TrajectoryResponse]],
        evaluation_results: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """Generate model performance analysis section."""
        performance_section = {
            'model_rankings': self._rank_models(evaluation_results),
            'metric_comparison': self._compare_metrics(evaluation_results),
            'prediction_quality': self._analyze_prediction_quality(predictions),
            'computational_performance': self._analyze_computational_performance(predictions),
            'robustness_analysis': self._analyze_model_robustness(predictions)
        }
        
        return performance_section
    
    def _generate_detailed_analysis(
        self,
        trajectories: List[TrajectoryData],
        predictions: Dict[str, List[TrajectoryResponse]],
        clustering_results: Optional[Dict],
        anomaly_results: Optional[Dict]
    ) -> Dict[str, Any]:
        """Generate detailed analysis section."""
        analysis = {
            'error_analysis': self._analyze_prediction_errors(predictions),
            'uncertainty_analysis': self._analyze_prediction_uncertainty(predictions),
            'failure_modes': self._identify_failure_modes(predictions),
        }
        
        if clustering_results:
            analysis['clustering_insights'] = self._analyze_clustering_insights(clustering_results)
        
        if anomaly_results:
            analysis['anomaly_insights'] = self._analyze_anomaly_insights(anomaly_results)
        
        return analysis
    
    def _generate_recommendations(
        self,
        evaluation_results: Dict[str, Dict[str, float]],
        clustering_results: Optional[Dict],
        anomaly_results: Optional[Dict]
    ) -> List[Dict[str, str]]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Model selection recommendations
        best_model = min(evaluation_results.keys(), 
                        key=lambda k: evaluation_results[k].get('rmse', float('inf')))
        
        recommendations.append({
            'category': 'Model Selection',
            'priority': 'High',
            'recommendation': f"Deploy {best_model} as primary model",
            'rationale': f"Shows lowest RMSE of {evaluation_results[best_model].get('rmse', 0):.3f}"
        })
        
        # Performance improvement recommendations
        avg_rmse = np.mean([metrics.get('rmse', 0) for metrics in evaluation_results.values()])
        if avg_rmse > 2.0:  # Threshold for acceptable performance
            recommendations.append({
                'category': 'Performance',
                'priority': 'Medium',
                'recommendation': "Investigate feature engineering improvements",
                'rationale': f"Average RMSE of {avg_rmse:.3f} exceeds acceptable threshold"
            })
        
        # Data quality recommendations
        if clustering_results and 'anomaly_analysis' in clustering_results:
            anomaly_rate = clustering_results['anomaly_analysis'].get('anomaly_rate', 0)
            if anomaly_rate > 0.1:  # More than 10% anomalies
                recommendations.append({
                    'category': 'Data Quality',
                    'priority': 'Medium',
                    'recommendation': "Implement anomaly detection in data pipeline",
                    'rationale': f"High anomaly rate detected: {anomaly_rate:.1%}"
                })
        
        # Monitoring recommendations
        recommendations.append({
            'category': 'Monitoring',
            'priority': 'High',
            'recommendation': "Implement continuous model monitoring",
            'rationale': "Essential for production deployment and drift detection"
        })
        
        return recommendations
    
    def _generate_appendix(
        self,
        trajectories: List[TrajectoryData],
        predictions: Dict[str, List[TrajectoryResponse]]
    ) -> Dict[str, Any]:
        """Generate appendix with technical details."""
        appendix = {
            'technical_specifications': {
                'data_format': "TrajectoryData schema with positions, velocities, timestamps",
                'coordinate_system': "Cartesian (x, y) coordinates in meters",
                'time_resolution': "Variable timestamp resolution",
                'feature_extraction': "Physics-informed kinematic features"
            },
            'model_configurations': self._extract_model_configurations(predictions),
            'evaluation_methodology': {
                'metrics_used': ['RMSE', 'MAE', 'ADE', 'FDE'],
                'validation_method': 'Time-series cross-validation',
                'prediction_horizon': 'Variable (model-dependent)'
            },
            'data_sources': {
                'dataset': "NGSIM trajectory dataset",
                'preprocessing': "Normalization, feature extraction, quality filtering"
            }
        }
        
        return appendix
    
    def _export_html_report(self, report: Dict[str, Any], output_path: str) -> str:
        """Export report as HTML."""
        html_content = self._generate_html_content(report)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(output_file)
    
    def _export_json_report(self, report: Dict[str, Any], output_path: str) -> str:
        """Export report as JSON."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        json_report = self._convert_numpy_types(report)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_report, f, indent=2, default=str)
        
        return str(output_file)
    
    def _generate_html_content(self, report: Dict[str, Any]) -> str:
        """Generate HTML content for the report."""
        metadata = report['metadata']
        
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{metadata['title']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 20px; }}
                .section {{ margin: 30px 0; }}
                .subsection {{ margin: 20px 0; }}
                h1 {{ color: #333; }}
                h2 {{ color: #555; border-bottom: 1px solid #ccc; padding-bottom: 10px; }}
                h3 {{ color: #777; }}
                .metric-table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                .metric-table th, .metric-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .metric-table th {{ background-color: #f2f2f2; }}
                .recommendation {{ background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-left: 4px solid #4CAF50; }}
                .finding {{ background-color: #e7f3ff; padding: 10px; margin: 5px 0; border-radius: 5px; }}
                .footer {{ margin-top: 50px; text-align: center; color: #888; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{metadata['title']}</h1>
                <p>Generated by: {metadata['author']}</p>
                <p>Date: {datetime.fromisoformat(metadata['generated_at']).strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            {self._generate_executive_summary_html(report['executive_summary'])}
            {self._generate_data_overview_html(report['data_overview'])}
            {self._generate_model_performance_html(report['model_performance'])}
            {self._generate_recommendations_html(report['recommendations'])}
            
            <div class="footer">
                <p>This report was automatically generated by the Trajectory Prediction System</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _generate_executive_summary_html(self, summary: Dict[str, Any]) -> str:
        """Generate HTML for executive summary."""
        findings_html = '\n'.join([f'<div class="finding">• {finding}</div>' 
                                 for finding in summary.get('key_findings', [])])
        
        metrics_rows = '\n'.join([
            f'<tr><td>{metric.upper()}</td><td>{value:.3f}</td></tr>'
            for metric, value in summary.get('performance_overview', {}).items()
        ])
        
        return f"""
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="subsection">
                <h3>Key Findings</h3>
                {findings_html}
            </div>
            <div class="subsection">
                <h3>Performance Overview</h3>
                <table class="metric-table">
                    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
                    <tbody>{metrics_rows}</tbody>
                </table>
            </div>
            <div class="subsection">
                <h3>Data Quality Score</h3>
                <p>Overall Score: {summary.get('data_quality_score', 0):.1%}</p>
            </div>
        </div>
        """
    
    def _generate_data_overview_html(self, overview: Dict[str, Any]) -> str:
        """Generate HTML for data overview."""
        if 'error' in overview:
            return f'<div class="section"><h2>Data Overview</h2><p>Error: {overview["error"]}</p></div>'
        
        stats = overview.get('dataset_statistics', {})
        stats_rows = '\n'.join([
            f'<tr><td>{key.replace("_", " ").title()}</td><td>{value:.2f}</td></tr>'
            for key, value in stats.items()
        ])
        
        return f"""
        <div class="section">
            <h2>Data Overview</h2>
            <div class="subsection">
                <h3>Dataset Statistics</h3>
                <table class="metric-table">
                    <thead><tr><th>Metric</th><th>Value</th></tr></thead>
                    <tbody>{stats_rows}</tbody>
                </table>
            </div>
        </div>
        """
    
    def _generate_model_performance_html(self, performance: Dict[str, Any]) -> str:
        """Generate HTML for model performance."""
        rankings = performance.get('model_rankings', [])
        ranking_rows = '\n'.join([
            f'<tr><td>{i+1}</td><td>{model["name"]}</td><td>{model["score"]:.3f}</td></tr>'
            for i, model in enumerate(rankings)
        ])
        
        return f"""
        <div class="section">
            <h2>Model Performance Analysis</h2>
            <div class="subsection">
                <h3>Model Rankings</h3>
                <table class="metric-table">
                    <thead><tr><th>Rank</th><th>Model</th><th>RMSE</th></tr></thead>
                    <tbody>{ranking_rows}</tbody>
                </table>
            </div>
        </div>
        """
    
    def _generate_recommendations_html(self, recommendations: List[Dict[str, str]]) -> str:
        """Generate HTML for recommendations."""
        rec_html = '\n'.join([
            f"""
            <div class="recommendation">
                <h4>{rec['category']} - {rec['priority']} Priority</h4>
                <p><strong>Recommendation:</strong> {rec['recommendation']}</p>
                <p><strong>Rationale:</strong> {rec['rationale']}</p>
            </div>
            """
            for rec in recommendations
        ])
        
        return f"""
        <div class="section">
            <h2>Recommendations</h2>
            {rec_html}
        </div>
        """
    
    # Helper methods
    def _calculate_avg_prediction_horizon(self, predictions: Dict[str, List[TrajectoryResponse]]) -> float:
        """Calculate average prediction horizon."""
        horizons = []
        for model_preds in predictions.values():
            for pred in model_preds:
                if pred.predicted_trajectory.timestamps:
                    horizon = pred.predicted_trajectory.timestamps[-1] - pred.predicted_trajectory.timestamps[0]
                    horizons.append(horizon)
        return np.mean(horizons) if horizons else 0
    
    def _calculate_data_quality_score(self, trajectories: List[TrajectoryData]) -> float:
        """Calculate overall data quality score."""
        if not trajectories:
            return 0.0
        
        scores = []
        for traj in trajectories:
            score = 1.0
            
            # Penalize missing data
            if not traj.positions:
                score -= 0.5
            if not traj.velocities:
                score -= 0.2
            if not traj.timestamps:
                score -= 0.2
            
            # Penalize short trajectories
            if len(traj.positions) < 10:
                score -= 0.1
            
            scores.append(max(0, score))
        
        return np.mean(scores)
    
    def _calculate_spatial_range(self, trajectories: List[TrajectoryData]) -> Dict[str, float]:
        """Calculate spatial coverage range."""
        x_coords = []
        y_coords = []
        
        for traj in trajectories:
            x_coords.extend([p.x for p in traj.positions])
            y_coords.extend([p.y for p in traj.positions])
        
        if not x_coords:
            return {}
        
        return {
            'x_min': min(x_coords),
            'x_max': max(x_coords),
            'y_min': min(y_coords),
            'y_max': max(y_coords),
            'x_range': max(x_coords) - min(x_coords),
            'y_range': max(y_coords) - min(y_coords)
        }
    
    def _calculate_temporal_range(self, trajectories: List[TrajectoryData]) -> Dict[str, float]:
        """Calculate temporal coverage range."""
        timestamps = []
        for traj in trajectories:
            timestamps.extend(traj.timestamps)
        
        if not timestamps:
            return {}
        
        return {
            'start_time': min(timestamps),
            'end_time': max(timestamps),
            'total_duration': max(timestamps) - min(timestamps)
        }
    
    def _calculate_speed_distribution(self, speeds: List[float]) -> Dict[str, float]:
        """Calculate speed distribution statistics."""
        if not speeds:
            return {}
        
        return {
            'mean': np.mean(speeds),
            'std': np.std(speeds),
            'min': min(speeds),
            'max': max(speeds),
            'median': np.median(speeds),
            'p95': np.percentile(speeds, 95)
        }
    
    def _is_complete_trajectory(self, trajectory: TrajectoryData) -> bool:
        """Check if trajectory is complete."""
        return (len(trajectory.positions) > 5 and 
                len(trajectory.timestamps) > 5 and
                trajectory.velocities is not None)
    
    def _calculate_missing_data_rate(self, trajectories: List[TrajectoryData]) -> float:
        """Calculate missing data rate."""
        total_fields = 0
        missing_fields = 0
        
        for traj in trajectories:
            total_fields += 3  # positions, velocities, timestamps
            
            if not traj.positions:
                missing_fields += 1
            if not traj.velocities:
                missing_fields += 1
            if not traj.timestamps:
                missing_fields += 1
        
        return missing_fields / total_fields if total_fields > 0 else 0
    
    def _calculate_outlier_rate(self, trajectories: List[TrajectoryData]) -> float:
        """Calculate outlier rate based on speed."""
        speeds = []
        for traj in trajectories:
            if traj.velocities:
                speeds.extend([v.magnitude for v in traj.velocities])
        
        if not speeds:
            return 0
        
        Q1 = np.percentile(speeds, 25)
        Q3 = np.percentile(speeds, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = sum(1 for speed in speeds if speed < lower_bound or speed > upper_bound)
        return outliers / len(speeds)
    
    def _rank_models(self, evaluation_results: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Rank models by performance."""
        rankings = []
        for model_name, metrics in evaluation_results.items():
            rmse = metrics.get('rmse', float('inf'))
            rankings.append({
                'name': model_name,
                'score': rmse,
                'metrics': metrics
            })
        
        return sorted(rankings, key=lambda x: x['score'])
    
    def _compare_metrics(self, evaluation_results: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Compare metrics across models."""
        metrics_comparison = {}
        
        for metric_name in ['rmse', 'mae', 'ade', 'fde']:
            values = {}
            for model_name, metrics in evaluation_results.items():
                values[model_name] = metrics.get(metric_name, 0)
            
            if values:
                metrics_comparison[metric_name] = {
                    'best_model': min(values, key=values.get),
                    'best_score': min(values.values()),
                    'worst_model': max(values, key=values.get),
                    'worst_score': max(values.values()),
                    'average': np.mean(list(values.values())),
                    'std': np.std(list(values.values()))
                }
        
        return metrics_comparison
    
    def _analyze_prediction_quality(self, predictions: Dict[str, List[TrajectoryResponse]]) -> Dict[str, Any]:
        """Analyze prediction quality across models."""
        quality_analysis = {}
        
        for model_name, model_preds in predictions.items():
            confidences = [pred.confidence for pred in model_preds]
            
            quality_analysis[model_name] = {
                'avg_confidence': np.mean(confidences),
                'confidence_std': np.std(confidences),
                'high_confidence_rate': sum(1 for c in confidences if c > 0.8) / len(confidences)
            }
        
        return quality_analysis
    
    def _analyze_computational_performance(self, predictions: Dict[str, List[TrajectoryResponse]]) -> Dict[str, Any]:
        """Analyze computational performance."""
        # This would need timing data from actual predictions
        # For now, return placeholder analysis
        return {
            'note': 'Computational performance analysis requires timing data from prediction execution'
        }
    
    def _analyze_model_robustness(self, predictions: Dict[str, List[TrajectoryResponse]]) -> Dict[str, Any]:
        """Analyze model robustness."""
        robustness = {}
        
        for model_name, model_preds in predictions.items():
            uncertainties = []
            for pred in model_preds:
                if pred.uncertainty:
                    avg_uncertainty = np.mean(list(pred.uncertainty.values()))
                    uncertainties.append(avg_uncertainty)
            
            robustness[model_name] = {
                'avg_uncertainty': np.mean(uncertainties) if uncertainties else 0,
                'uncertainty_consistency': 1.0 / (1.0 + np.std(uncertainties)) if uncertainties else 0
            }
        
        return robustness
    
    def _analyze_prediction_errors(self, predictions: Dict[str, List[TrajectoryResponse]]) -> Dict[str, Any]:
        """Analyze prediction error patterns."""
        # This would require ground truth data for comparison
        return {
            'note': 'Error analysis requires ground truth data for comparison'
        }
    
    def _analyze_prediction_uncertainty(self, predictions: Dict[str, List[TrajectoryResponse]]) -> Dict[str, Any]:
        """Analyze prediction uncertainty patterns."""
        uncertainty_analysis = {}
        
        for model_name, model_preds in predictions.items():
            uncertainties = []
            for pred in model_preds:
                if pred.uncertainty:
                    uncertainties.extend(list(pred.uncertainty.values()))
            
            if uncertainties:
                uncertainty_analysis[model_name] = {
                    'mean_uncertainty': np.mean(uncertainties),
                    'uncertainty_range': max(uncertainties) - min(uncertainties),
                    'uncertainty_distribution': {
                        'p25': np.percentile(uncertainties, 25),
                        'p50': np.percentile(uncertainties, 50),
                        'p75': np.percentile(uncertainties, 75),
                        'p95': np.percentile(uncertainties, 95)
                    }
                }
        
        return uncertainty_analysis
    
    def _identify_failure_modes(self, predictions: Dict[str, List[TrajectoryResponse]]) -> List[Dict[str, str]]:
        """Identify common failure modes."""
        failure_modes = []
        
        # Check for low confidence predictions
        for model_name, model_preds in predictions.items():
            low_confidence_count = sum(1 for pred in model_preds if pred.confidence < 0.5)
            if low_confidence_count > len(model_preds) * 0.2:  # More than 20%
                failure_modes.append({
                    'model': model_name,
                    'mode': 'Low Confidence Predictions',
                    'description': f'{low_confidence_count} predictions with confidence < 50%'
                })
        
        return failure_modes
    
    def _analyze_clustering_insights(self, clustering_results: Dict) -> Dict[str, Any]:
        """Analyze clustering results for insights."""
        if not clustering_results:
            return {}
        
        insights = {
            'cluster_quality': clustering_results.get('quality_metrics', {}),
            'dominant_patterns': clustering_results.get('cluster_analysis', {}),
            'pattern_diversity': len(clustering_results.get('cluster_labels', []))
        }
        
        return insights
    
    def _analyze_anomaly_insights(self, anomaly_results: Dict) -> Dict[str, Any]:
        """Analyze anomaly detection results for insights."""
        if not anomaly_results:
            return {}
        
        anomaly_analysis = anomaly_results.get('anomaly_analysis', {})
        
        insights = {
            'anomaly_rate': anomaly_analysis.get('anomaly_rate', 0),
            'detection_quality': 'Good' if anomaly_analysis.get('anomaly_rate', 0) < 0.1 else 'Review needed',
            'anomaly_characteristics': anomaly_analysis.get('anomaly_characteristics', {})
        }
        
        return insights
    
    def _extract_model_configurations(self, predictions: Dict[str, List[TrajectoryResponse]]) -> Dict[str, Dict[str, Any]]:
        """Extract model configuration information."""
        configurations = {}
        
        for model_name in predictions.keys():
            # This would ideally extract actual model configurations
            # For now, return placeholder information
            configurations[model_name] = {
                'model_type': model_name.split('_')[0] if '_' in model_name else model_name,
                'features': 'Physics-informed trajectory features',
                'hyperparameters': 'Default configuration'
            }
        
        return configurations
    
    def _convert_numpy_types(self, obj: Any) -> Any:
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj


class DataQualityReporter:
    """Specialized reporter for data quality assessment."""
    
    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
    
    def generate_data_quality_report(
        self,
        trajectories: List[TrajectoryData]
    ) -> Dict[str, Any]:
        """Generate comprehensive data quality report."""
        report = {
            'summary': self._generate_quality_summary(trajectories),
            'completeness_analysis': self._analyze_completeness(trajectories),
            'consistency_analysis': self._analyze_consistency(trajectories),
            'accuracy_analysis': self._analyze_accuracy(trajectories),
            'timeliness_analysis': self._analyze_timeliness(trajectories),
            'recommendations': self._generate_quality_recommendations(trajectories)
        }
        
        return report
    
    def _generate_quality_summary(self, trajectories: List[TrajectoryData]) -> Dict[str, Any]:
        """Generate overall quality summary."""
        if not trajectories:
            return {'error': 'No trajectory data to analyze'}
        
        # Calculate quality scores
        completeness_score = self._calculate_completeness_score(trajectories)
        consistency_score = self._calculate_consistency_score(trajectories)
        accuracy_score = self._calculate_accuracy_score(trajectories)
        
        overall_score = (completeness_score + consistency_score + accuracy_score) / 3
        
        return {
            'overall_quality_score': overall_score,
            'completeness_score': completeness_score,
            'consistency_score': consistency_score,
            'accuracy_score': accuracy_score,
            'total_trajectories': len(trajectories),
            'quality_grade': self._get_quality_grade(overall_score)
        }
    
    def _analyze_completeness(self, trajectories: List[TrajectoryData]) -> Dict[str, Any]:
        """Analyze data completeness."""
        complete_trajectories = 0
        missing_positions = 0
        missing_velocities = 0
        missing_timestamps = 0
        
        for traj in trajectories:
            is_complete = True
            
            if not traj.positions or len(traj.positions) == 0:
                missing_positions += 1
                is_complete = False
            
            if not traj.velocities or len(traj.velocities) == 0:
                missing_velocities += 1
                is_complete = False
            
            if not traj.timestamps or len(traj.timestamps) == 0:
                missing_timestamps += 1
                is_complete = False
            
            if is_complete:
                complete_trajectories += 1
        
        total = len(trajectories)
        
        return {
            'complete_trajectories': complete_trajectories,
            'completeness_rate': complete_trajectories / total if total > 0 else 0,
            'missing_data': {
                'positions': missing_positions,
                'velocities': missing_velocities,
                'timestamps': missing_timestamps
            },
            'missing_rates': {
                'positions': missing_positions / total if total > 0 else 0,
                'velocities': missing_velocities / total if total > 0 else 0,
                'timestamps': missing_timestamps / total if total > 0 else 0
            }
        }
    
    def _analyze_consistency(self, trajectories: List[TrajectoryData]) -> Dict[str, Any]:
        """Analyze data consistency."""
        inconsistent_length = 0
        inconsistent_timing = 0
        inconsistent_physics = 0
        
        for traj in trajectories:
            # Check length consistency
            if traj.positions and traj.velocities and traj.timestamps:
                pos_len = len(traj.positions)
                vel_len = len(traj.velocities)
                time_len = len(traj.timestamps)
                
                if not (pos_len == vel_len == time_len):
                    inconsistent_length += 1
            
            # Check timing consistency
            if traj.timestamps and len(traj.timestamps) > 1:
                time_diffs = np.diff(traj.timestamps)
                if np.any(time_diffs <= 0):  # Non-monotonic timestamps
                    inconsistent_timing += 1
            
            # Check physics consistency (basic)
            if traj.positions and traj.velocities and len(traj.positions) > 1:
                # Check if velocities are reasonable given position changes
                for i in range(1, min(len(traj.positions), len(traj.velocities))):
                    pos_change = np.sqrt(
                        (traj.positions[i].x - traj.positions[i-1].x)**2 + 
                        (traj.positions[i].y - traj.positions[i-1].y)**2
                    )
                    velocity = traj.velocities[i].magnitude
                    if velocity > 0 and pos_change / velocity > 10:  # Unreasonable ratio
                        inconsistent_physics += 1
                        break
        
        total = len(trajectories)
        
        return {
            'consistent_trajectories': total - max(inconsistent_length, inconsistent_timing, inconsistent_physics),
            'inconsistency_issues': {
                'length_mismatch': inconsistent_length,
                'timing_errors': inconsistent_timing,
                'physics_violations': inconsistent_physics
            },
            'consistency_rates': {
                'length_consistency': 1 - (inconsistent_length / total) if total > 0 else 1,
                'timing_consistency': 1 - (inconsistent_timing / total) if total > 0 else 1,
                'physics_consistency': 1 - (inconsistent_physics / total) if total > 0 else 1
            }
        }
    
    def _analyze_accuracy(self, trajectories: List[TrajectoryData]) -> Dict[str, Any]:
        """Analyze data accuracy."""
        # This is a simplified accuracy analysis
        # In practice, you'd need ground truth or reference data
        
        outlier_trajectories = 0
        unrealistic_speeds = 0
        unrealistic_accelerations = 0
        
        for traj in trajectories:
            # Check for unrealistic speeds
            if traj.velocities:
                max_speed = max(v.magnitude for v in traj.velocities)
                if max_speed > 60:  # > 60 m/s (216 km/h) seems unrealistic for most scenarios
                    unrealistic_speeds += 1
            
            # Check for unrealistic accelerations
            if traj.velocities and len(traj.velocities) > 1:
                accelerations = []
                for i in range(1, len(traj.velocities)):
                    if traj.timestamps and i < len(traj.timestamps):
                        dt = traj.timestamps[i] - traj.timestamps[i-1]
                        if dt > 0:
                            dv = traj.velocities[i].magnitude - traj.velocities[i-1].magnitude
                            acc = abs(dv / dt)
                            accelerations.append(acc)
                
                if accelerations and max(accelerations) > 10:  # > 10 m/s² seems high
                    unrealistic_accelerations += 1
        
        # Outliers are trajectories with either unrealistic speeds or accelerations
        outlier_trajectories = len(set().union(
            {i for i, traj in enumerate(trajectories) 
             if traj.velocities and max(v.magnitude for v in traj.velocities) > 60},
            {i for i, traj in enumerate(trajectories) 
             if unrealistic_accelerations > 0}
        ))
        
        total = len(trajectories)
        
        return {
            'accurate_trajectories': total - outlier_trajectories,
            'accuracy_rate': (total - outlier_trajectories) / total if total > 0 else 0,
            'accuracy_issues': {
                'unrealistic_speeds': unrealistic_speeds,
                'unrealistic_accelerations': unrealistic_accelerations,
                'total_outliers': outlier_trajectories
            }
        }
    
    def _analyze_timeliness(self, trajectories: List[TrajectoryData]) -> Dict[str, Any]:
        """Analyze data timeliness."""
        # This would typically compare against data collection timestamps
        # For now, analyze trajectory duration consistency
        
        durations = []
        for traj in trajectories:
            if traj.timestamps and len(traj.timestamps) > 1:
                duration = traj.timestamps[-1] - traj.timestamps[0]
                durations.append(duration)
        
        if not durations:
            return {'error': 'No timing data available for timeliness analysis'}
        
        return {
            'average_duration': np.mean(durations),
            'duration_std': np.std(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'duration_consistency': 1.0 / (1.0 + np.std(durations) / np.mean(durations)) if np.mean(durations) > 0 else 0
        }
    
    def _calculate_completeness_score(self, trajectories: List[TrajectoryData]) -> float:
        """Calculate completeness score (0-1)."""
        if not trajectories:
            return 0.0
        
        complete_count = sum(
            1 for traj in trajectories 
            if traj.positions and traj.velocities and traj.timestamps and
               len(traj.positions) > 0 and len(traj.velocities) > 0 and len(traj.timestamps) > 0
        )
        
        return complete_count / len(trajectories)
    
    def _calculate_consistency_score(self, trajectories: List[TrajectoryData]) -> float:
        """Calculate consistency score (0-1)."""
        if not trajectories:
            return 0.0
        
        consistent_count = 0
        for traj in trajectories:
            if (traj.positions and traj.velocities and traj.timestamps and
                len(traj.positions) == len(traj.velocities) == len(traj.timestamps)):
                # Check timestamp ordering
                if len(traj.timestamps) <= 1 or all(
                    traj.timestamps[i] < traj.timestamps[i+1] 
                    for i in range(len(traj.timestamps)-1)
                ):
                    consistent_count += 1
        
        return consistent_count / len(trajectories)
    
    def _calculate_accuracy_score(self, trajectories: List[TrajectoryData]) -> float:
        """Calculate accuracy score (0-1) based on outlier detection."""
        if not trajectories:
            return 0.0
        
        outlier_count = 0
        for traj in trajectories:
            if traj.velocities:
                speeds = [v.magnitude for v in traj.velocities]
                # Simple outlier detection based on speed
                if speeds and (max(speeds) > 60 or min(speeds) < 0):
                    outlier_count += 1
        
        return (len(trajectories) - outlier_count) / len(trajectories)
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to grade."""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.8:
            return "Good"
        elif score >= 0.7:
            return "Fair"
        elif score >= 0.6:
            return "Poor"
        else:
            return "Critical"
    
    def _generate_quality_recommendations(self, trajectories: List[TrajectoryData]) -> List[Dict[str, str]]:
        """Generate data quality improvement recommendations."""
        recommendations = []
        
        # Analyze current quality issues
        completeness = self._analyze_completeness(trajectories)
        consistency = self._analyze_consistency(trajectories)
        accuracy = self._analyze_accuracy(trajectories)
        
        # Completeness recommendations
        if completeness['completeness_rate'] < 0.9:
            recommendations.append({
                'category': 'Data Completeness',
                'priority': 'High',
                'recommendation': 'Implement data validation at collection stage',
                'rationale': f"Only {completeness['completeness_rate']:.1%} of trajectories are complete"
            })
        
        # Consistency recommendations
        if consistency['consistency_rates']['length_consistency'] < 0.95:
            recommendations.append({
                'category': 'Data Consistency',
                'priority': 'Medium',
                'recommendation': 'Add length validation checks',
                'rationale': 'Inconsistent array lengths detected across trajectory components'
            })
        
        # Accuracy recommendations
        if accuracy['accuracy_rate'] < 0.9:
            recommendations.append({
                'category': 'Data Accuracy',
                'priority': 'Medium',
                'recommendation': 'Implement outlier detection and filtering',
                'rationale': f"Outliers detected in {accuracy['accuracy_issues']['total_outliers']} trajectories"
            })
        
        return recommendations


class ModelComparisonReporter:
    """Specialized reporter for model comparison analysis."""
    
    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
    
    def generate_comparison_report(
        self,
        predictions: Dict[str, List[TrajectoryResponse]],
        evaluation_results: Dict[str, Dict[str, float]],
        ground_truth: Optional[List[TrajectoryData]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive model comparison report."""
        report = {
            'executive_summary': self._generate_comparison_summary(evaluation_results),
            'performance_analysis': self._analyze_model_performance(evaluation_results),
            'prediction_quality': self._analyze_prediction_quality(predictions),
            'statistical_significance': self._test_statistical_significance(evaluation_results),
            'recommendation_engine': self._generate_model_recommendations(
                predictions, evaluation_results
            )
        }
        
        return report
    
    def _generate_comparison_summary(self, evaluation_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Generate comparison summary."""
        if not evaluation_results:
            return {'error': 'No evaluation results available'}
        
        # Find best and worst models
        rmse_scores = {model: metrics.get('rmse', float('inf')) 
                      for model, metrics in evaluation_results.items()}
        
        best_model = min(rmse_scores, key=rmse_scores.get)
        worst_model = max(rmse_scores, key=rmse_scores.get)
        
        # Calculate performance spread
        performance_spread = max(rmse_scores.values()) - min(rmse_scores.values())
        
        return {
            'models_compared': len(evaluation_results),
            'best_performer': {
                'model': best_model,
                'rmse': rmse_scores[best_model]
            },
            'worst_performer': {
                'model': worst_model,
                'rmse': rmse_scores[worst_model]
            },
            'performance_spread': performance_spread,
            'performance_consistency': 1.0 / (1.0 + np.std(list(rmse_scores.values()))),
            'deployment_ready': rmse_scores[best_model] < 2.0  # Arbitrary threshold
        }
    
    def _analyze_model_performance(self, evaluation_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Analyze detailed model performance."""
        analysis = {
            'metric_rankings': {},
            'performance_profiles': {},
            'consistency_analysis': {}
        }
        
        # Rank models by each metric
        for metric in ['rmse', 'mae', 'ade', 'fde']:
            if any(metric in metrics for metrics in evaluation_results.values()):
                scores = {model: metrics.get(metric, float('inf')) 
                         for model, metrics in evaluation_results.items()}
                
                analysis['metric_rankings'][metric] = sorted(
                    scores.items(), key=lambda x: x[1]
                )
        
        # Create performance profiles
        for model, metrics in evaluation_results.items():
            analysis['performance_profiles'][model] = {
                'strengths': [],
                'weaknesses': [],
                'overall_rank': self._calculate_overall_rank(model, evaluation_results)
            }
            
            # Identify strengths and weaknesses
            for metric, value in metrics.items():
                all_values = [m.get(metric, float('inf')) for m in evaluation_results.values()]
                if value == min(all_values):
                    analysis['performance_profiles'][model]['strengths'].append(metric)
                elif value == max(all_values):
                    analysis['performance_profiles'][model]['weaknesses'].append(metric)
        
        return analysis
    
    def _analyze_prediction_quality(self, predictions: Dict[str, List[TrajectoryResponse]]) -> Dict[str, Any]:
        """Analyze prediction quality aspects."""
        quality_analysis = {}
        
        for model_name, model_preds in predictions.items():
            confidences = [pred.confidence for pred in model_preds if pred.confidence is not None]
            
            # Uncertainty analysis
            uncertainties = []
            for pred in model_preds:
                if pred.uncertainty:
                    uncertainties.extend(pred.uncertainty.values())
            
            quality_analysis[model_name] = {
                'confidence_metrics': {
                    'mean': np.mean(confidences) if confidences else 0,
                    'std': np.std(confidences) if confidences else 0,
                    'min': min(confidences) if confidences else 0,
                    'max': max(confidences) if confidences else 0
                },
                'uncertainty_metrics': {
                    'mean': np.mean(uncertainties) if uncertainties else 0,
                    'consistency': 1.0 / (1.0 + np.std(uncertainties)) if uncertainties else 0
                },
                'prediction_stability': self._calculate_stability(model_preds)
            }
        
        return quality_analysis
    
    def _test_statistical_significance(self, evaluation_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Test statistical significance of performance differences."""
        # This is simplified - in practice you'd need multiple runs or bootstrap sampling
        significance_tests = {}
        
        models = list(evaluation_results.keys())
        
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                rmse1 = evaluation_results[model1].get('rmse', 0)
                rmse2 = evaluation_results[model2].get('rmse', 0)
                
                # Simple difference test (would need proper statistical test in practice)
                difference = abs(rmse1 - rmse2)
                relative_difference = difference / max(rmse1, rmse2) if max(rmse1, rmse2) > 0 else 0
                
                significance_tests[f"{model1}_vs_{model2}"] = {
                    'absolute_difference': difference,
                    'relative_difference': relative_difference,
                    'significant': relative_difference > 0.05,  # 5% threshold
                    'better_model': model1 if rmse1 < rmse2 else model2
                }
        
        return significance_tests
    
    def _generate_model_recommendations(
        self,
        predictions: Dict[str, List[TrajectoryResponse]],
        evaluation_results: Dict[str, Dict[str, float]]
    ) -> List[Dict[str, str]]:
        """Generate model selection and deployment recommendations."""
        recommendations = []
        
        # Find best overall model
        rmse_scores = {model: metrics.get('rmse', float('inf')) 
                      for model, metrics in evaluation_results.items()}
        best_model = min(rmse_scores, key=rmse_scores.get)
        
        # Primary deployment recommendation
        recommendations.append({
            'category': 'Primary Deployment',
            'priority': 'High',
            'recommendation': f'Deploy {best_model} as primary model',
            'rationale': f'Best RMSE performance: {rmse_scores[best_model]:.3f}'
        })
        
        # Ensemble recommendation
        top_models = sorted(rmse_scores.items(), key=lambda x: x[1])[:3]
        if len(top_models) > 1 and top_models[1][1] - top_models[0][1] < 0.1:
            recommendations.append({
                'category': 'Ensemble Strategy',
                'priority': 'Medium',
                'recommendation': f'Consider ensemble of top models: {[m[0] for m in top_models[:3]]}',
                'rationale': 'Multiple models show similar performance'
            })
        
        # Specialized use case recommendations
        for model, metrics in evaluation_results.items():
            if metrics.get('fde', float('inf')) < min(
                m.get('fde', float('inf')) for m in evaluation_results.values()
            ):
                recommendations.append({
                    'category': 'Specialized Use',
                    'priority': 'Low',
                    'recommendation': f'Use {model} for long-horizon predictions',
                    'rationale': 'Best Final Displacement Error performance'
                })
        
        return recommendations
    
    def _calculate_overall_rank(self, model: str, evaluation_results: Dict[str, Dict[str, float]]) -> int:
        """Calculate overall ranking for a model."""
        # Simple ranking based on RMSE
        rmse_scores = {m: metrics.get('rmse', float('inf')) 
                      for m, metrics in evaluation_results.items()}
        
        sorted_models = sorted(rmse_scores.items(), key=lambda x: x[1])
        
        for i, (model_name, _) in enumerate(sorted_models):
            if model_name == model:
                return i + 1  # 1-indexed ranking
        
        return len(sorted_models)
    
    def _calculate_stability(self, predictions: List[TrajectoryResponse]) -> float:
        """Calculate prediction stability score."""
        if not predictions:
            return 0.0
        
        # Simple stability based on confidence variance
        confidences = [pred.confidence for pred in predictions if pred.confidence is not None]
        
        if not confidences:
            return 0.0
        
        # Stability = 1 / (1 + coefficient of variation)
        mean_confidence = np.mean(confidences)
        std_confidence = np.std(confidences)
        
        if mean_confidence == 0:
            return 0.0
        
        cv = std_confidence / mean_confidence
        return 1.0 / (1.0 + cv)