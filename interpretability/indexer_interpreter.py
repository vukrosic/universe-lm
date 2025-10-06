#!/usr/bin/env python3
"""
Lightning Indexer interpretability analysis
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import json
from collections import defaultdict

class IndexerInterpreter:
    """Interprets Lightning Indexer behavior and patterns"""
    
    def __init__(self, save_dir: str = "indexer_analysis"):
        self.save_dir = save_dir
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
    def analyze_indexer_heads(self, model, batch: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze what each indexer head focuses on
        
        Args:
            model: Model with Lightning Indexer
            batch: Input batch [batch_size, seq_len, d_model]
            
        Returns:
            Dictionary with head analysis results
        """
        if not hasattr(model, 'indexer'):
            raise ValueError("Model must have an 'indexer' attribute")
            
        # Get indexer scores
        with torch.no_grad():
            indexer_scores = model.indexer(batch)
            
        if indexer_scores.is_cuda:
            indexer_scores = indexer_scores.cpu()
            
        batch_size, n_heads, seq_len, _ = indexer_scores.shape
        
        head_analysis = {}
        
        for head_idx in range(n_heads):
            head_scores = indexer_scores[:, head_idx, :, :]  # [batch, seq_len, seq_len]
            
            # Analyze head patterns
            head_analysis[head_idx] = self._analyze_single_head(head_scores, head_idx)
            
        return head_analysis
    
    def _analyze_single_head(self, head_scores: torch.Tensor, head_idx: int) -> Dict[str, Any]:
        """Analyze patterns for a single indexer head"""
        batch_size, seq_len, _ = head_scores.shape
        
        # Convert to numpy
        head_scores_np = head_scores.numpy()
        
        # 1. Score distribution
        all_scores = head_scores_np.flatten()
        score_stats = {
            'mean': np.mean(all_scores),
            'std': np.std(all_scores),
            'min': np.min(all_scores),
            'max': np.max(all_scores),
            'median': np.median(all_scores)
        }
        
        # 2. Attention pattern analysis
        # Average across batch
        avg_scores = np.mean(head_scores_np, axis=0)  # [seq_len, seq_len]
        
        # Analyze attention patterns
        pattern_analysis = self._analyze_attention_patterns(avg_scores)
        
        # 3. Positional bias analysis
        positional_bias = self._analyze_positional_bias(head_scores_np)
        
        # 4. Token selection analysis
        token_selection = self._analyze_token_selection(head_scores_np)
        
        return {
            'head_idx': head_idx,
            'score_stats': score_stats,
            'pattern_analysis': pattern_analysis,
            'positional_bias': positional_bias,
            'token_selection': token_selection
        }
    
    def _analyze_attention_patterns(self, attention_scores: np.ndarray) -> Dict[str, Any]:
        """Analyze attention patterns in indexer scores"""
        seq_len = attention_scores.shape[0]
        
        # 1. Local vs global patterns
        local_scores = 0
        global_scores = 0
        total_scores = 0
        
        for i in range(seq_len):
            for j in range(seq_len):
                if abs(i - j) <= 5:  # Local window
                    local_scores += attention_scores[i, j]
                else:
                    global_scores += attention_scores[i, j]
                total_scores += attention_scores[i, j]
        
        local_ratio = local_scores / total_scores if total_scores > 0 else 0
        global_ratio = global_scores / total_scores if total_scores > 0 else 0
        
        # 2. Diagonal vs off-diagonal
        diagonal_scores = np.sum(np.diag(attention_scores))
        off_diagonal_scores = np.sum(attention_scores) - diagonal_scores
        diagonal_ratio = diagonal_scores / (off_diagonal_scores + 1e-8)
        
        # 3. Sparsity analysis
        threshold = np.percentile(attention_scores, 90)  # Top 10% scores
        sparse_ratio = np.sum(attention_scores < threshold) / (seq_len * seq_len)
        
        # 4. Concentration analysis
        # How concentrated are the scores
        concentration = np.std(attention_scores)
        
        return {
            'local_ratio': local_ratio,
            'global_ratio': global_ratio,
            'diagonal_ratio': diagonal_ratio,
            'sparse_ratio': sparse_ratio,
            'concentration': concentration
        }
    
    def _analyze_positional_bias(self, head_scores: np.ndarray) -> Dict[str, Any]:
        """Analyze positional bias in indexer scores"""
        batch_size, seq_len, _ = head_scores.shape
        
        # Analyze query position bias
        query_bias = []
        for pos in range(seq_len):
            pos_scores = head_scores[:, pos, :].mean(axis=0)  # Average across batch
            query_bias.append(np.mean(pos_scores))
        
        # Analyze key position bias
        key_bias = []
        for pos in range(seq_len):
            pos_scores = head_scores[:, :, pos].mean(axis=0)  # Average across batch
            key_bias.append(np.mean(pos_scores))
        
        # Calculate bias statistics
        query_bias_std = np.std(query_bias)
        key_bias_std = np.std(key_bias)
        
        # Check for specific biases
        early_bias = np.mean(query_bias[:seq_len//4]) - np.mean(query_bias[seq_len//4:])
        late_bias = np.mean(query_bias[3*seq_len//4:]) - np.mean(query_bias[:3*seq_len//4])
        
        return {
            'query_bias': query_bias,
            'key_bias': key_bias,
            'query_bias_std': query_bias_std,
            'key_bias_std': key_bias_std,
            'early_bias': early_bias,
            'late_bias': late_bias
        }
    
    def _analyze_token_selection(self, head_scores: np.ndarray) -> Dict[str, Any]:
        """Analyze token selection patterns"""
        batch_size, seq_len, _ = head_scores.shape
        
        # For each query position, find top-k selected tokens
        top_k = min(10, seq_len // 2)
        
        selection_patterns = []
        for batch_idx in range(batch_size):
            for query_pos in range(seq_len):
                scores = head_scores[batch_idx, query_pos, :]
                top_indices = np.argsort(scores)[-top_k:]
                selection_patterns.append(top_indices)
        
        # Analyze selection patterns
        selection_patterns = np.array(selection_patterns)
        
        # 1. Position preference
        position_preference = np.zeros(seq_len)
        for pattern in selection_patterns:
            for pos in pattern:
                position_preference[pos] += 1
        position_preference = position_preference / len(selection_patterns)
        
        # 2. Distance analysis
        distances = []
        for pattern in selection_patterns:
            for pos in pattern:
                distances.append(pos)
        
        distance_stats = {
            'mean': np.mean(distances),
            'std': np.std(distances),
            'min': np.min(distances),
            'max': np.max(distances)
        }
        
        return {
            'position_preference': position_preference,
            'distance_stats': distance_stats,
            'selection_patterns': selection_patterns
        }
    
    def study_relevance_distribution(self, model, dataset) -> Dict[str, Any]:
        """
        Study relevance score distribution across dataset
        
        Args:
            model: Model with Lightning Indexer
            dataset: Dataset to analyze
            
        Returns:
            Dictionary with distribution analysis
        """
        all_scores = []
        all_selected_tokens = []
        
        with torch.no_grad():
            for batch in dataset:
                # Get indexer scores
                indexer_scores = model.indexer(batch)
                
                if indexer_scores.is_cuda:
                    indexer_scores = indexer_scores.cpu()
                
                # Get selected tokens
                if hasattr(model, 'get_selected_tokens'):
                    selected_tokens = model.get_selected_tokens(batch)
                    if selected_tokens.is_cuda:
                        selected_tokens = selected_tokens.cpu()
                    all_selected_tokens.append(selected_tokens)
                
                all_scores.append(indexer_scores)
        
        # Concatenate all scores
        all_scores = torch.cat(all_scores, dim=0)
        if all_selected_tokens:
            all_selected_tokens = torch.cat(all_selected_tokens, dim=0)
        
        # Analyze distribution
        distribution_analysis = self._analyze_score_distribution(all_scores, all_selected_tokens)
        
        return distribution_analysis
    
    def _analyze_score_distribution(self, all_scores: torch.Tensor, 
                                  all_selected_tokens: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Analyze distribution of relevance scores"""
        batch_size, n_heads, seq_len, _ = all_scores.shape
        
        # Flatten scores
        flat_scores = all_scores.flatten().numpy()
        
        # Calculate distribution statistics
        score_stats = {
            'mean': np.mean(flat_scores),
            'std': np.std(flat_scores),
            'min': np.min(flat_scores),
            'max': np.max(flat_scores),
            'median': np.median(flat_scores),
            'percentiles': {
                '25th': np.percentile(flat_scores, 25),
                '75th': np.percentile(flat_scores, 75),
                '90th': np.percentile(flat_scores, 90),
                '95th': np.percentile(flat_scores, 95),
                '99th': np.percentile(flat_scores, 99)
            }
        }
        
        # Analyze head-wise distributions
        head_distributions = {}
        for head_idx in range(n_heads):
            head_scores = all_scores[:, head_idx, :, :].flatten().numpy()
            head_distributions[head_idx] = {
                'mean': np.mean(head_scores),
                'std': np.std(head_scores),
                'min': np.min(head_scores),
                'max': np.max(head_scores)
            }
        
        # Analyze selected vs non-selected tokens
        selection_analysis = None
        if all_selected_tokens is not None:
            selection_analysis = self._analyze_selection_vs_scores(all_scores, all_selected_tokens)
        
        return {
            'overall_stats': score_stats,
            'head_distributions': head_distributions,
            'selection_analysis': selection_analysis
        }
    
    def _analyze_selection_vs_scores(self, all_scores: torch.Tensor, 
                                   all_selected_tokens: torch.Tensor) -> Dict[str, Any]:
        """Analyze relationship between scores and selection"""
        batch_size, n_heads, seq_len, _ = all_scores.shape
        
        selected_scores = []
        non_selected_scores = []
        
        for batch_idx in range(batch_size):
            for head_idx in range(n_heads):
                scores = all_scores[batch_idx, head_idx, :, :]
                selected = all_selected_tokens[batch_idx, :]
                
                # Get scores for selected and non-selected tokens
                for query_pos in range(seq_len):
                    if selected[query_pos]:
                        selected_scores.extend(scores[query_pos, :].tolist())
                    else:
                        non_selected_scores.extend(scores[query_pos, :].tolist())
        
        # Calculate statistics
        selected_stats = {
            'mean': np.mean(selected_scores),
            'std': np.std(selected_scores),
            'min': np.min(selected_scores),
            'max': np.max(selected_scores)
        }
        
        non_selected_stats = {
            'mean': np.mean(non_selected_scores),
            'std': np.std(non_selected_scores),
            'min': np.min(non_selected_scores),
            'max': np.max(non_selected_scores)
        }
        
        # Calculate separation
        separation = selected_stats['mean'] - non_selected_stats['mean']
        
        return {
            'selected_stats': selected_stats,
            'non_selected_stats': non_selected_stats,
            'separation': separation
        }
    
    def visualize_indexer_analysis(self, head_analysis: Dict[str, Any],
                                 title: str = "Indexer Head Analysis",
                                 save_path: str = None) -> None:
        """Visualize indexer head analysis results"""
        n_heads = len(head_analysis)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Score statistics by head
        ax1 = axes[0, 0]
        head_indices = list(head_analysis.keys())
        means = [head_analysis[head_idx]['score_stats']['mean'] for head_idx in head_indices]
        stds = [head_analysis[head_idx]['score_stats']['std'] for head_idx in head_indices]
        
        ax1.bar(head_indices, means, yerr=stds, alpha=0.7, color=self.colors[:n_heads])
        ax1.set_title('Score Statistics by Head')
        ax1.set_xlabel('Head Index')
        ax1.set_ylabel('Mean Score')
        
        # 2. Pattern analysis
        ax2 = axes[0, 1]
        local_ratios = [head_analysis[head_idx]['pattern_analysis']['local_ratio'] for head_idx in head_indices]
        global_ratios = [head_analysis[head_idx]['pattern_analysis']['global_ratio'] for head_idx in head_indices]
        
        x = np.arange(n_heads)
        width = 0.35
        
        ax2.bar(x - width/2, local_ratios, width, label='Local', alpha=0.7)
        ax2.bar(x + width/2, global_ratios, width, label='Global', alpha=0.7)
        ax2.set_title('Local vs Global Patterns')
        ax2.set_xlabel('Head Index')
        ax2.set_ylabel('Ratio')
        ax2.legend()
        
        # 3. Positional bias
        ax3 = axes[1, 0]
        query_biases = [head_analysis[head_idx]['positional_bias']['query_bias_std'] for head_idx in head_indices]
        key_biases = [head_analysis[head_idx]['positional_bias']['key_bias_std'] for head_idx in head_indices]
        
        ax3.bar(x - width/2, query_biases, width, label='Query Bias', alpha=0.7)
        ax3.bar(x + width/2, key_biases, width, label='Key Bias', alpha=0.7)
        ax3.set_title('Positional Bias by Head')
        ax3.set_xlabel('Head Index')
        ax3.set_ylabel('Bias Std')
        ax3.legend()
        
        # 4. Token selection patterns
        ax4 = axes[1, 1]
        position_preferences = [head_analysis[head_idx]['token_selection']['position_preference'] for head_idx in head_indices]
        
        for head_idx, preferences in enumerate(position_preferences):
            ax4.plot(preferences, label=f'Head {head_idx}', alpha=0.7)
        ax4.set_title('Position Preference Patterns')
        ax4.set_xlabel('Position')
        ax4.set_ylabel('Selection Frequency')
        ax4.legend()
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_analysis(self, analysis_results: Dict[str, Any], filepath: str) -> None:
        """Save indexer analysis results"""
        # Convert tensors to lists for JSON serialization
        serializable_data = {}
        for key, value in analysis_results.items():
            if isinstance(value, torch.Tensor):
                serializable_data[key] = value.cpu().numpy().tolist()
            elif isinstance(value, np.ndarray):
                serializable_data[key] = value.tolist()
            else:
                serializable_data[key] = value
                
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2)
    
    def load_analysis(self, filepath: str) -> Dict[str, Any]:
        """Load indexer analysis results"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Convert lists back to tensors where appropriate
        tensor_data = {}
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], list):
                    tensor_data[key] = torch.tensor(value)
                else:
                    tensor_data[key] = torch.tensor(value)
            else:
                tensor_data[key] = value
                
        return tensor_data

def demo_indexer_interpretation():
    """Demo function showing how to use IndexerInterpreter"""
    print("🔍 Indexer Interpretation Demo")
    print("=" * 40)
    
    # Create a mock model with indexer
    class MockModel:
        def __init__(self, d_model=64, n_heads=4, seq_len=32):
            self.d_model = d_model
            self.n_heads = n_heads
            self.seq_len = seq_len
            
        def indexer(self, batch):
            # Mock indexer that returns random scores
            batch_size = batch.shape[0]
            scores = torch.randn(batch_size, self.n_heads, self.seq_len, self.seq_len)
            return torch.softmax(scores, dim=-1)
    
    # Create sample data
    batch_size, seq_len, d_model = 2, 32, 64
    batch = torch.randn(batch_size, seq_len, d_model)
    
    # Create model and interpreter
    model = MockModel(d_model, n_heads=4, seq_len=seq_len)
    interpreter = IndexerInterpreter()
    
    # Analyze indexer heads
    head_analysis = interpreter.analyze_indexer_heads(model, batch)
    
    print("Head Analysis Results:")
    for head_idx, analysis in head_analysis.items():
        print(f"\nHead {head_idx}:")
        print(f"  Score Stats: {analysis['score_stats']}")
        print(f"  Pattern Analysis: {analysis['pattern_analysis']}")
        print(f"  Positional Bias: {analysis['positional_bias']['query_bias_std']:.4f}")
    
    # Visualize analysis
    interpreter.visualize_indexer_analysis(head_analysis, title="Demo Indexer Analysis")
    
    print("\n✅ Demo completed! Use IndexerInterpreter to understand your Lightning Indexer.")

if __name__ == "__main__":
    demo_indexer_interpretation()
