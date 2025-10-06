#!/usr/bin/env python3
"""
Attention pattern visualization for sparse attention interpretability
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import json

class AttentionVisualizer:
    """Visualizes attention patterns in sparse attention models"""
    
    def __init__(self, save_dir: str = "visualizations"):
        self.save_dir = save_dir
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
    def visualize_sparse_attention(self, attention_weights: torch.Tensor, 
                                 selected_tokens: torch.Tensor = None,
                                 title: str = "Sparse Attention Pattern",
                                 save_path: str = None) -> None:
        """
        Visualize sparse attention pattern
        
        Args:
            attention_weights: [batch, heads, seq_len, seq_len] attention weights
            selected_tokens: [batch, seq_len] boolean mask of selected tokens
            title: Plot title
            save_path: Path to save visualization
        """
        # Convert to numpy for visualization
        if attention_weights.is_cuda:
            attention_weights = attention_weights.cpu()
        if selected_tokens is not None and selected_tokens.is_cuda:
            selected_tokens = selected_tokens.cpu()
            
        batch_size, n_heads, seq_len, _ = attention_weights.shape
        
        # Create subplots for each head
        fig, axes = plt.subplots(2, (n_heads + 1) // 2, figsize=(4 * n_heads, 8))
        if n_heads == 1:
            axes = [axes]
        elif n_heads <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
            
        for head_idx in range(n_heads):
            ax = axes[head_idx]
            
            # Get attention weights for this head (average across batch)
            head_weights = attention_weights[:, head_idx, :, :].mean(dim=0).numpy()
            
            # Create heatmap
            sns.heatmap(head_weights, ax=ax, cmap='Blues', cbar=True)
            ax.set_title(f'Head {head_idx}')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            
            # Highlight selected tokens if provided
            if selected_tokens is not None:
                selected_mask = selected_tokens.mean(dim=0).numpy()
                for i, is_selected in enumerate(selected_mask):
                    if is_selected:
                        ax.axhline(y=i, color='red', alpha=0.3, linewidth=2)
                        ax.axvline(x=i, color='red', alpha=0.3, linewidth=2)
        
        # Hide unused subplots
        for idx in range(n_heads, len(axes)):
            axes[idx].set_visible(False)
            
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def compare_sparse_vs_dense(self, sparse_weights: torch.Tensor,
                               dense_weights: torch.Tensor,
                               selected_tokens: torch.Tensor = None,
                               title: str = "Sparse vs Dense Attention",
                               save_path: str = None) -> None:
        """
        Compare sparse and dense attention patterns
        
        Args:
            sparse_weights: Sparse attention weights
            dense_weights: Dense attention weights
            selected_tokens: Selected tokens mask
            title: Plot title
            save_path: Path to save visualization
        """
        # Convert to numpy
        if sparse_weights.is_cuda:
            sparse_weights = sparse_weights.cpu()
        if dense_weights.is_cuda:
            dense_weights = dense_weights.cpu()
        if selected_tokens is not None and selected_tokens.is_cuda:
            selected_tokens = selected_tokens.cpu()
            
        batch_size, n_heads, seq_len, _ = sparse_weights.shape
        
        # Create comparison plot
        fig, axes = plt.subplots(2, n_heads, figsize=(5 * n_heads, 10))
        if n_heads == 1:
            axes = axes.reshape(2, 1)
            
        for head_idx in range(n_heads):
            # Sparse attention
            ax_sparse = axes[0, head_idx]
            sparse_head = sparse_weights[:, head_idx, :, :].mean(dim=0).numpy()
            sns.heatmap(sparse_head, ax=ax_sparse, cmap='Blues', cbar=True)
            ax_sparse.set_title(f'Sparse Head {head_idx}')
            ax_sparse.set_xlabel('Key Position')
            ax_sparse.set_ylabel('Query Position')
            
            # Dense attention
            ax_dense = axes[1, head_idx]
            dense_head = dense_weights[:, head_idx, :, :].mean(dim=0).numpy()
            sns.heatmap(dense_head, ax=ax_dense, cmap='Blues', cbar=True)
            ax_dense.set_title(f'Dense Head {head_idx}')
            ax_dense.set_xlabel('Key Position')
            ax_dense.set_ylabel('Query Position')
            
            # Highlight selected tokens
            if selected_tokens is not None:
                selected_mask = selected_tokens.mean(dim=0).numpy()
                for i, is_selected in enumerate(selected_mask):
                    if is_selected:
                        ax_sparse.axhline(y=i, color='red', alpha=0.3, linewidth=2)
                        ax_sparse.axvline(x=i, color='red', alpha=0.3, linewidth=2)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def visualize_token_selection(self, indexer_scores: torch.Tensor,
                                selected_tokens: torch.Tensor,
                                top_k: int,
                                title: str = "Token Selection Analysis",
                                save_path: str = None) -> None:
        """
        Visualize token selection process
        
        Args:
            indexer_scores: [batch, seq_len, seq_len] indexer scores
            selected_tokens: [batch, seq_len] selected tokens mask
            top_k: Number of selected tokens
            title: Plot title
            save_path: Path to save visualization
        """
        # Convert to numpy
        if indexer_scores.is_cuda:
            indexer_scores = indexer_scores.cpu()
        if selected_tokens.is_cuda:
            selected_tokens = selected_tokens.cpu()
            
        batch_size, seq_len, _ = indexer_scores.shape
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Indexer scores heatmap
        ax1 = axes[0, 0]
        scores_avg = indexer_scores.mean(dim=0).numpy()
        sns.heatmap(scores_avg, ax=ax1, cmap='viridis', cbar=True)
        ax1.set_title('Indexer Scores (Average)')
        ax1.set_xlabel('Key Position')
        ax1.set_ylabel('Query Position')
        
        # 2. Selected tokens distribution
        ax2 = axes[0, 1]
        selected_dist = selected_tokens.mean(dim=0).numpy()
        ax2.bar(range(seq_len), selected_dist, color='skyblue', alpha=0.7)
        ax2.set_title(f'Token Selection Distribution (Top-{top_k})')
        ax2.set_xlabel('Token Position')
        ax2.set_ylabel('Selection Frequency')
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='50% threshold')
        ax2.legend()
        
        # 3. Score distribution
        ax3 = axes[1, 0]
        all_scores = indexer_scores.flatten().numpy()
        ax3.hist(all_scores, bins=50, alpha=0.7, color='lightgreen')
        ax3.set_title('Indexer Score Distribution')
        ax3.set_xlabel('Score Value')
        ax3.set_ylabel('Frequency')
        
        # 4. Selection efficiency
        ax4 = axes[1, 1]
        selection_efficiency = []
        for i in range(seq_len):
            # Calculate how often position i is selected
            selection_rate = selected_tokens[:, i].float().mean().item()
            selection_efficiency.append(selection_rate)
        
        ax4.plot(range(seq_len), selection_efficiency, 'o-', color='orange')
        ax4.set_title('Selection Efficiency by Position')
        ax4.set_xlabel('Token Position')
        ax4.set_ylabel('Selection Rate')
        ax4.axhline(y=top_k/seq_len, color='red', linestyle='--', alpha=0.5, 
                   label=f'Expected rate ({top_k/seq_len:.2f})')
        ax4.legend()
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def analyze_attention_entropy(self, attention_weights: torch.Tensor,
                                title: str = "Attention Entropy Analysis",
                                save_path: str = None) -> Dict[str, float]:
        """
        Analyze attention entropy patterns
        
        Args:
            attention_weights: [batch, heads, seq_len, seq_len] attention weights
            title: Plot title
            save_path: Path to save visualization
            
        Returns:
            Dictionary with entropy statistics
        """
        # Convert to numpy
        if attention_weights.is_cuda:
            attention_weights = attention_weights.cpu()
            
        batch_size, n_heads, seq_len, _ = attention_weights.shape
        
        # Calculate entropy for each head
        entropies = []
        for head_idx in range(n_heads):
            head_weights = attention_weights[:, head_idx, :, :]  # [batch, seq_len, seq_len]
            
            # Calculate entropy for each query position
            head_entropies = []
            for batch_idx in range(batch_size):
                for query_idx in range(seq_len):
                    # Get attention distribution for this query
                    attn_dist = head_weights[batch_idx, query_idx, :]
                    
                    # Calculate entropy
                    entropy = -torch.sum(attn_dist * torch.log(attn_dist + 1e-8))
                    head_entropies.append(entropy.item())
            
            entropies.append(head_entropies)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Entropy distribution by head
        ax1 = axes[0, 0]
        for head_idx, head_entropies in enumerate(entropies):
            ax1.hist(head_entropies, bins=30, alpha=0.6, 
                    label=f'Head {head_idx}', color=self.colors[head_idx % len(self.colors)])
        ax1.set_title('Attention Entropy Distribution by Head')
        ax1.set_xlabel('Entropy')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # 2. Average entropy by head
        ax2 = axes[0, 1]
        avg_entropies = [np.mean(head_entropies) for head_entropies in entropies]
        ax2.bar(range(n_heads), avg_entropies, color=self.colors[:n_heads])
        ax2.set_title('Average Entropy by Head')
        ax2.set_xlabel('Head Index')
        ax2.set_ylabel('Average Entropy')
        
        # 3. Entropy vs position
        ax3 = axes[1, 0]
        for head_idx, head_entropies in enumerate(entropies):
            # Reshape to [batch, seq_len] and average across batch
            head_entropies_reshaped = np.array(head_entropies).reshape(batch_size, seq_len)
            avg_entropy_by_pos = head_entropies_reshaped.mean(axis=0)
            ax3.plot(range(seq_len), avg_entropy_by_pos, 'o-', 
                    label=f'Head {head_idx}', color=self.colors[head_idx % len(self.colors)])
        ax3.set_title('Entropy vs Query Position')
        ax3.set_xlabel('Query Position')
        ax3.set_ylabel('Average Entropy')
        ax3.legend()
        
        # 4. Entropy statistics
        ax4 = axes[1, 1]
        all_entropies = np.concatenate(entropies)
        ax4.hist(all_entropies, bins=50, alpha=0.7, color='lightblue')
        ax4.axvline(np.mean(all_entropies), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_entropies):.3f}')
        ax4.axvline(np.median(all_entropies), color='green', linestyle='--', 
                   label=f'Median: {np.median(all_entropies):.3f}')
        ax4.set_title('Overall Entropy Distribution')
        ax4.set_xlabel('Entropy')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Return statistics
        stats = {
            'mean_entropy': np.mean(all_entropies),
            'median_entropy': np.median(all_entropies),
            'std_entropy': np.std(all_entropies),
            'min_entropy': np.min(all_entropies),
            'max_entropy': np.max(all_entropies),
            'head_avg_entropies': avg_entropies
        }
        
        return stats
        
    def visualize_pattern_evolution(self, attention_patterns: Dict[int, torch.Tensor],
                                  title: str = "Attention Pattern Evolution",
                                  save_path: str = None) -> None:
        """
        Visualize how attention patterns evolve with sequence length
        
        Args:
            attention_patterns: Dictionary mapping sequence length to attention weights
            title: Plot title
            save_path: Path to save visualization
        """
        seq_lengths = sorted(attention_patterns.keys())
        n_lengths = len(seq_lengths)
        
        # Create visualization
        fig, axes = plt.subplots(2, (n_lengths + 1) // 2, figsize=(5 * n_lengths, 10))
        if n_lengths == 1:
            axes = [axes]
        elif n_lengths <= 2:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
            
        for idx, seq_len in enumerate(seq_lengths):
            ax = axes[idx]
            
            # Get attention weights for this sequence length
            attention_weights = attention_patterns[seq_len]
            if attention_weights.is_cuda:
                attention_weights = attention_weights.cpu()
                
            # Average across batch and heads
            avg_weights = attention_weights.mean(dim=(0, 1)).numpy()
            
            # Create heatmap
            sns.heatmap(avg_weights, ax=ax, cmap='Blues', cbar=True)
            ax.set_title(f'Sequence Length {seq_len}')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            
        # Hide unused subplots
        for idx in range(n_lengths, len(axes)):
            axes[idx].set_visible(False)
            
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_visualization_data(self, data: Dict[str, Any], filepath: str) -> None:
        """Save visualization data to JSON file"""
        # Convert tensors to lists for JSON serialization
        serializable_data = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                serializable_data[key] = value.cpu().numpy().tolist()
            elif isinstance(value, np.ndarray):
                serializable_data[key] = value.tolist()
            else:
                serializable_data[key] = value
                
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2)
            
    def load_visualization_data(self, filepath: str) -> Dict[str, Any]:
        """Load visualization data from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Convert lists back to tensors
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

def demo_attention_visualization():
    """Demo function showing how to use AttentionVisualizer"""
    print("🎨 Attention Visualization Demo")
    print("=" * 40)
    
    # Create sample data
    batch_size, n_heads, seq_len = 2, 4, 64
    attention_weights = torch.randn(batch_size, n_heads, seq_len, seq_len)
    attention_weights = torch.softmax(attention_weights, dim=-1)
    
    # Create selected tokens mask
    selected_tokens = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    for i in range(batch_size):
        # Select random tokens
        selected_indices = torch.randperm(seq_len)[:seq_len//2]
        selected_tokens[i, selected_indices] = True
    
    # Create visualizer
    visualizer = AttentionVisualizer()
    
    # Visualize sparse attention
    visualizer.visualize_sparse_attention(
        attention_weights, 
        selected_tokens,
        title="Demo Sparse Attention Pattern"
    )
    
    # Analyze entropy
    entropy_stats = visualizer.analyze_attention_entropy(
        attention_weights,
        title="Demo Attention Entropy Analysis"
    )
    
    print("Entropy Statistics:")
    for key, value in entropy_stats.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✅ Demo completed! Use AttentionVisualizer to analyze your models.")

if __name__ == "__main__":
    demo_attention_visualization()
