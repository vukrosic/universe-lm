#!/usr/bin/env python3
"""
Pattern analysis for sparse attention interpretability
"""

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from typing import Dict, List, Tuple, Any, Optional
import json
from collections import defaultdict

class PatternAnalyzer:
    """Analyzes learned attention patterns in sparse attention models"""
    
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.cluster_centers_ = None
        self.cluster_labels_ = None
        self.pattern_features_ = None
        
    def extract_pattern_features(self, attention_weights: torch.Tensor) -> np.ndarray:
        """
        Extract features from attention patterns for clustering
        
        Args:
            attention_weights: [batch, heads, seq_len, seq_len] attention weights
            
        Returns:
            Feature matrix [n_patterns, n_features]
        """
        if attention_weights.is_cuda:
            attention_weights = attention_weights.cpu()
            
        batch_size, n_heads, seq_len, _ = attention_weights.shape
        
        features = []
        
        for batch_idx in range(batch_size):
            for head_idx in range(n_heads):
                # Get attention pattern for this head
                pattern = attention_weights[batch_idx, head_idx, :, :].numpy()
                
                # Extract features
                pattern_features = self._extract_single_pattern_features(pattern)
                features.append(pattern_features)
                
        return np.array(features)
    
    def _extract_single_pattern_features(self, pattern: np.ndarray) -> np.ndarray:
        """Extract features from a single attention pattern"""
        seq_len = pattern.shape[0]
        
        features = []
        
        # 1. Sparsity features
        sparsity_ratio = np.sum(pattern < 0.01) / (seq_len * seq_len)
        features.append(sparsity_ratio)
        
        # 2. Entropy features
        entropy = -np.sum(pattern * np.log(pattern + 1e-8))
        features.append(entropy)
        
        # 3. Concentration features
        max_attention = np.max(pattern, axis=1)
        concentration = np.mean(max_attention)
        features.append(concentration)
        
        # 4. Local vs global features
        local_attention = np.sum(pattern * np.eye(seq_len))
        global_attention = np.sum(pattern * (1 - np.eye(seq_len)))
        local_global_ratio = local_attention / (global_attention + 1e-8)
        features.append(local_global_ratio)
        
        # 5. Positional bias features
        # Check if attention is biased towards certain positions
        row_sums = np.sum(pattern, axis=1)
        col_sums = np.sum(pattern, axis=0)
        
        # Positional bias (higher values for positions with more attention)
        positional_bias_row = np.std(row_sums)
        positional_bias_col = np.std(col_sums)
        features.extend([positional_bias_row, positional_bias_col])
        
        # 6. Diagonal vs off-diagonal features
        diagonal_sum = np.sum(np.diag(pattern))
        off_diagonal_sum = np.sum(pattern) - diagonal_sum
        diagonal_ratio = diagonal_sum / (off_diagonal_sum + 1e-8)
        features.append(diagonal_ratio)
        
        # 7. Attention spread features
        # How spread out is the attention
        attention_spread = np.std(pattern)
        features.append(attention_spread)
        
        # 8. Top-k attention features
        # How much attention goes to top-k tokens
        for k in [5, 10, 20]:
            if k < seq_len:
                top_k_indices = np.argsort(pattern, axis=1)[:, -k:]
                top_k_attention = np.sum(pattern[np.arange(seq_len)[:, None], top_k_indices])
                top_k_ratio = top_k_attention / np.sum(pattern)
                features.append(top_k_ratio)
            else:
                features.append(1.0)  # All attention is top-k
        
        return np.array(features)
    
    def cluster_patterns(self, attention_weights: torch.Tensor) -> Dict[str, Any]:
        """
        Cluster attention patterns into different types
        
        Args:
            attention_weights: [batch, heads, seq_len, seq_len] attention weights
            
        Returns:
            Dictionary with clustering results
        """
        # Extract features
        features = self.extract_pattern_features(attention_weights)
        self.pattern_features_ = features
        
        # Perform clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        cluster_labels = kmeans.fit_predict(features)
        
        self.cluster_centers_ = kmeans.cluster_centers_
        self.cluster_labels_ = cluster_labels
        
        # Calculate silhouette score
        silhouette_avg = silhouette_score(features, cluster_labels)
        
        # Analyze clusters
        cluster_analysis = self._analyze_clusters(features, cluster_labels)
        
        return {
            'cluster_labels': cluster_labels,
            'cluster_centers': kmeans.cluster_centers_,
            'silhouette_score': silhouette_avg,
            'cluster_analysis': cluster_analysis,
            'features': features
        }
    
    def _analyze_clusters(self, features: np.ndarray, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """Analyze characteristics of each cluster"""
        cluster_analysis = {}
        
        for cluster_id in range(self.n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_features = features[cluster_mask]
            
            if len(cluster_features) > 0:
                cluster_analysis[cluster_id] = {
                    'size': len(cluster_features),
                    'mean_features': np.mean(cluster_features, axis=0),
                    'std_features': np.std(cluster_features, axis=0),
                    'sparsity_ratio': np.mean(cluster_features[:, 0]),
                    'entropy': np.mean(cluster_features[:, 1]),
                    'concentration': np.mean(cluster_features[:, 2]),
                    'local_global_ratio': np.mean(cluster_features[:, 3])
                }
        
        return cluster_analysis
    
    def classify_pattern_types(self, attention_weights: torch.Tensor) -> Dict[str, List[int]]:
        """
        Classify attention patterns into different types
        
        Args:
            attention_weights: [batch, heads, seq_len, seq_len] attention weights
            
        Returns:
            Dictionary mapping pattern types to indices
        """
        if attention_weights.is_cuda:
            attention_weights = attention_weights.cpu()
            
        batch_size, n_heads, seq_len, _ = attention_weights.shape
        
        pattern_types = {
            'local': [],
            'global': [],
            'sparse': [],
            'dense': [],
            'diagonal': [],
            'off_diagonal': []
        }
        
        for batch_idx in range(batch_size):
            for head_idx in range(n_heads):
                pattern = attention_weights[batch_idx, head_idx, :, :].numpy()
                pattern_idx = batch_idx * n_heads + head_idx
                
                # Classify pattern type
                if self._is_local_pattern(pattern):
                    pattern_types['local'].append(pattern_idx)
                elif self._is_global_pattern(pattern):
                    pattern_types['global'].append(pattern_idx)
                elif self._is_sparse_pattern(pattern):
                    pattern_types['sparse'].append(pattern_idx)
                elif self._is_dense_pattern(pattern):
                    pattern_types['dense'].append(pattern_idx)
                elif self._is_diagonal_pattern(pattern):
                    pattern_types['diagonal'].append(pattern_idx)
                else:
                    pattern_types['off_diagonal'].append(pattern_idx)
        
        return pattern_types
    
    def _is_local_pattern(self, pattern: np.ndarray, window_size: int = 5) -> bool:
        """Check if pattern is local (attends to nearby tokens)"""
        seq_len = pattern.shape[0]
        local_attention = 0
        total_attention = 0
        
        for i in range(seq_len):
            for j in range(seq_len):
                if abs(i - j) <= window_size:
                    local_attention += pattern[i, j]
                total_attention += pattern[i, j]
        
        return (local_attention / total_attention) > 0.7
    
    def _is_global_pattern(self, pattern: np.ndarray) -> bool:
        """Check if pattern is global (attends to distant tokens)"""
        seq_len = pattern.shape[0]
        global_attention = 0
        total_attention = 0
        
        for i in range(seq_len):
            for j in range(seq_len):
                if abs(i - j) > 5:
                    global_attention += pattern[i, j]
                total_attention += pattern[i, j]
        
        return (global_attention / total_attention) > 0.6
    
    def _is_sparse_pattern(self, pattern: np.ndarray, threshold: float = 0.01) -> bool:
        """Check if pattern is sparse (many low attention weights)"""
        sparse_ratio = np.sum(pattern < threshold) / (pattern.shape[0] * pattern.shape[1])
        return sparse_ratio > 0.8
    
    def _is_dense_pattern(self, pattern: np.ndarray, threshold: float = 0.01) -> bool:
        """Check if pattern is dense (few low attention weights)"""
        sparse_ratio = np.sum(pattern < threshold) / (pattern.shape[0] * pattern.shape[1])
        return sparse_ratio < 0.2
    
    def _is_diagonal_pattern(self, pattern: np.ndarray) -> bool:
        """Check if pattern is diagonal (attends to same position)"""
        diagonal_sum = np.sum(np.diag(pattern))
        total_sum = np.sum(pattern)
        return (diagonal_sum / total_sum) > 0.5
    
    def analyze_pattern_efficiency(self, attention_weights: torch.Tensor,
                                model_outputs: torch.Tensor) -> Dict[str, float]:
        """
        Analyze efficiency of different attention patterns
        
        Args:
            attention_weights: [batch, heads, seq_len, seq_len] attention weights
            model_outputs: [batch, seq_len, d_model] model outputs
            
        Returns:
            Dictionary with efficiency metrics
        """
        if attention_weights.is_cuda:
            attention_weights = attention_weights.cpu()
        if model_outputs.is_cuda:
            model_outputs = model_outputs.cpu()
            
        batch_size, n_heads, seq_len, _ = attention_weights.shape
        
        efficiency_metrics = {
            'sparsity_ratio': [],
            'attention_entropy': [],
            'computational_cost': [],
            'output_quality': [],
            'efficiency_score': []
        }
        
        for batch_idx in range(batch_size):
            for head_idx in range(n_heads):
                pattern = attention_weights[batch_idx, head_idx, :, :].numpy()
                output = model_outputs[batch_idx, :, :].numpy()
                
                # Calculate efficiency metrics
                sparsity_ratio = np.sum(pattern < 0.01) / (seq_len * seq_len)
                attention_entropy = -np.sum(pattern * np.log(pattern + 1e-8))
                computational_cost = seq_len * seq_len  # O(L²) for dense
                if sparsity_ratio > 0.5:
                    computational_cost = seq_len * int(seq_len * (1 - sparsity_ratio))  # O(Lk) for sparse
                
                # Simple output quality metric (variance in outputs)
                output_quality = np.var(output)
                
                # Efficiency score (quality per unit cost)
                efficiency_score = output_quality / computational_cost
                
                efficiency_metrics['sparsity_ratio'].append(sparsity_ratio)
                efficiency_metrics['attention_entropy'].append(attention_entropy)
                efficiency_metrics['computational_cost'].append(computational_cost)
                efficiency_metrics['output_quality'].append(output_quality)
                efficiency_metrics['efficiency_score'].append(efficiency_score)
        
        # Calculate summary statistics
        summary_stats = {}
        for metric, values in efficiency_metrics.items():
            summary_stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return summary_stats
    
    def identify_optimal_patterns(self, efficiency_metrics: Dict[str, float],
                                threshold: float = 0.8) -> List[int]:
        """
        Identify optimal patterns based on efficiency metrics
        
        Args:
            efficiency_metrics: Efficiency metrics from analyze_pattern_efficiency
            threshold: Threshold for considering a pattern optimal
            
        Returns:
            List of indices of optimal patterns
        """
        efficiency_scores = efficiency_metrics['efficiency_score']['mean']
        optimal_threshold = np.percentile(efficiency_scores, threshold * 100)
        
        optimal_patterns = []
        for i, score in enumerate(efficiency_scores):
            if score >= optimal_threshold:
                optimal_patterns.append(i)
        
        return optimal_patterns
    
    def save_analysis(self, filepath: str) -> None:
        """Save pattern analysis results"""
        analysis_data = {
            'cluster_centers': self.cluster_centers_.tolist() if self.cluster_centers_ is not None else None,
            'cluster_labels': self.cluster_labels_.tolist() if self.cluster_labels_ is not None else None,
            'pattern_features': self.pattern_features_.tolist() if self.pattern_features_ is not None else None,
            'n_clusters': self.n_clusters,
            'random_state': self.random_state
        }
        
        with open(filepath, 'w') as f:
            json.dump(analysis_data, f, indent=2)
    
    def load_analysis(self, filepath: str) -> None:
        """Load pattern analysis results"""
        with open(filepath, 'r') as f:
            analysis_data = json.load(f)
        
        self.n_clusters = analysis_data['n_clusters']
        self.random_state = analysis_data['random_state']
        
        if analysis_data['cluster_centers'] is not None:
            self.cluster_centers_ = np.array(analysis_data['cluster_centers'])
        if analysis_data['cluster_labels'] is not None:
            self.cluster_labels_ = np.array(analysis_data['cluster_labels'])
        if analysis_data['pattern_features'] is not None:
            self.pattern_features_ = np.array(analysis_data['pattern_features'])

def demo_pattern_analysis():
    """Demo function showing how to use PatternAnalyzer"""
    print("🔍 Pattern Analysis Demo")
    print("=" * 40)
    
    # Create sample data
    batch_size, n_heads, seq_len = 4, 4, 32
    attention_weights = torch.randn(batch_size, n_heads, seq_len, seq_len)
    attention_weights = torch.softmax(attention_weights, dim=-1)
    
    # Create sample model outputs
    d_model = 64
    model_outputs = torch.randn(batch_size, seq_len, d_model)
    
    # Create analyzer
    analyzer = PatternAnalyzer(n_clusters=3)
    
    # Cluster patterns
    clustering_results = analyzer.cluster_patterns(attention_weights)
    print(f"Silhouette Score: {clustering_results['silhouette_score']:.4f}")
    
    # Classify pattern types
    pattern_types = analyzer.classify_pattern_types(attention_weights)
    print("\nPattern Type Distribution:")
    for pattern_type, indices in pattern_types.items():
        print(f"  {pattern_type}: {len(indices)} patterns")
    
    # Analyze efficiency
    efficiency_metrics = analyzer.analyze_pattern_efficiency(attention_weights, model_outputs)
    print("\nEfficiency Metrics:")
    for metric, stats in efficiency_metrics.items():
        print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    # Identify optimal patterns
    optimal_patterns = analyzer.identify_optimal_patterns(efficiency_metrics)
    print(f"\nOptimal Patterns: {len(optimal_patterns)} out of {batch_size * n_heads}")
    
    print("\n✅ Demo completed! Use PatternAnalyzer to understand your attention patterns.")

if __name__ == "__main__":
    demo_pattern_analysis()
