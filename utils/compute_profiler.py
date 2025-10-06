#!/usr/bin/env python3
"""
Comprehensive compute profiler for sparse attention experiments
"""

import torch
import time
import psutil
import subprocess
from typing import Dict, Any, Callable
import json

class ComputeProfiler:
    """Profiles computational bottlenecks in sparse attention models"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.timings = {}
        self.memory_usage = {}
        self.flops_estimates = {}
        
    def time_component(self, name: str, func: Callable, *args, **kwargs) -> Any:
        """Time a component with CUDA events for accuracy"""
        if self.device == 'cuda' and torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            torch.cuda.synchronize()
            start_event.record()
            result = func(*args, **kwargs)
            end_event.record()
            torch.cuda.synchronize()
            
            time_taken = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
        else:
            start_time = time.time()
            result = func(*args, **kwargs)
            time_taken = time.time() - start_time
            
        self.timings[name] = time_taken
        return result
    
    def measure_memory(self, name: str, func: Callable, *args, **kwargs) -> Any:
        """Measure peak memory usage of a component"""
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            result = func(*args, **kwargs)
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        else:
            # CPU memory measurement
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024**3  # GB
            
            result = func(*args, **kwargs)
            
            memory_after = process.memory_info().rss / 1024**3  # GB
            peak_memory = memory_after - memory_before
            
        self.memory_usage[name] = peak_memory
        return result
    
    def estimate_flops(self, d_model: int, seq_len: int, n_heads: int, 
                       sparsity_ratio: float = 0.5, indexer_heads: int = 4, 
                       indexer_dim: int = 64) -> Dict[str, float]:
        """Estimate FLOPs for different components"""
        k = max(1, int(seq_len * sparsity_ratio))
        
        # QKV projection FLOPs
        qkv_flops = 3 * seq_len * d_model * d_model
        
        # Attention computation FLOPs (sparse)
        attention_flops = seq_len * k * d_model
        
        # Output projection FLOPs
        output_flops = seq_len * d_model * d_model
        
        # Lightning Indexer FLOPs (O(L²))
        indexer_flops = indexer_heads * seq_len * seq_len * indexer_dim
        
        # FFN FLOPs (assuming 4x expansion)
        ffn_flops = 2 * seq_len * d_model * (4 * d_model)
        
        total_flops = qkv_flops + attention_flops + output_flops + indexer_flops + ffn_flops
        
        flops_breakdown = {
            'total': total_flops,
            'qkv_projection': qkv_flops,
            'attention': attention_flops,
            'output_projection': output_flops,
            'indexer': indexer_flops,
            'ffn': ffn_flops,
            'indexer_ratio': indexer_flops / total_flops,
            'attention_ratio': attention_flops / total_flops,
            'sparsity_savings': (seq_len * seq_len * d_model - attention_flops) / (seq_len * seq_len * d_model)
        }
        
        self.flops_estimates = flops_breakdown
        return flops_breakdown
    
    def profile_sparse_attention(self, model, batch, seq_len: int, d_model: int, n_heads: int):
        """Profile all components of sparse attention"""
        results = {}
        
        # Profile Lightning Indexer
        def indexer_only():
            if hasattr(model, 'indexer'):
                return model.indexer(batch)
            return None
            
        results['indexer'] = self.time_component('indexer', indexer_only)
        
        # Profile Attention computation
        def attention_only():
            if hasattr(model, 'attention'):
                return model.attention(batch)
            return None
            
        results['attention'] = self.time_component('attention', attention_only)
        
        # Profile Full Forward Pass
        def full_forward():
            return model(batch)
            
        results['full'] = self.time_component('full', full_forward)
        
        # Estimate FLOPs
        flops = self.estimate_flops(d_model, seq_len, n_heads)
        
        return results, flops
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get current GPU statistics"""
        if not torch.cuda.is_available():
            return {}
            
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                parts = [p.strip() for p in result.stdout.strip().split(',')]
                if len(parts) >= 4:
                    return {
                        'gpu_utilization': int(parts[0]),
                        'memory_used': int(parts[1]),
                        'memory_total': int(parts[2]),
                        'temperature': int(parts[3])
                    }
        except Exception as e:
            print(f"Error getting GPU stats: {e}")
            
        return {}
    
    def print_report(self, save_to_file: str = None):
        """Print comprehensive profiling report"""
        report = {
            'timings': self.timings,
            'memory_usage': self.memory_usage,
            'flops_estimates': self.flops_estimates,
            'gpu_stats': self.get_gpu_stats()
        }
        
        print("🔍 Compute Profile Report")
        print("=" * 60)
        
        # Timing breakdown
        if self.timings:
            print("\n⏱️  Timing Breakdown:")
            total_time = sum(self.timings.values())
            for name, time_taken in sorted(self.timings.items(), key=lambda x: x[1], reverse=True):
                percentage = (time_taken / total_time) * 100
                print(f"  {name:20}: {time_taken:.4f}s ({percentage:5.1f}%)")
            print(f"  {'Total':20}: {total_time:.4f}s")
        
        # Memory breakdown
        if self.memory_usage:
            print("\n💾 Memory Usage:")
            for name, memory in sorted(self.memory_usage.items(), key=lambda x: x[1], reverse=True):
                print(f"  {name:20}: {memory:.2f} GB")
        
        # FLOPs breakdown
        if self.flops_estimates:
            print("\n🧮 FLOPs Breakdown:")
            total_flops = self.flops_estimates['total']
            for name, flops in self.flops_estimates.items():
                if name != 'total' and not name.endswith('_ratio') and not name.endswith('_savings'):
                    percentage = (flops / total_flops) * 100
                    print(f"  {name:20}: {flops/1e9:.2f} GFLOPs ({percentage:5.1f}%)")
            
            print(f"\n  Indexer overhead: {self.flops_estimates['indexer_ratio']*100:.1f}%")
            print(f"  Sparsity savings: {self.flops_estimates['sparsity_savings']*100:.1f}%")
        
        # GPU stats
        gpu_stats = self.get_gpu_stats()
        if gpu_stats:
            print("\n🖥️  GPU Status:")
            print(f"  Utilization: {gpu_stats['gpu_utilization']}%")
            print(f"  Memory: {gpu_stats['memory_used']}/{gpu_stats['memory_total']} MB")
            print(f"  Temperature: {gpu_stats['temperature']}°C")
        
        # Save to file if requested
        if save_to_file:
            with open(save_to_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"\n💾 Report saved to: {save_to_file}")
    
    def identify_bottlenecks(self) -> Dict[str, str]:
        """Identify the main computational bottlenecks"""
        bottlenecks = {}
        
        if self.timings:
            total_time = sum(self.timings.values())
            max_component = max(self.timings.items(), key=lambda x: x[1])
            bottlenecks['time_bottleneck'] = f"{max_component[0]} ({max_component[1]/total_time*100:.1f}%)"
        
        if self.flops_estimates:
            if self.flops_estimates['indexer_ratio'] > 0.3:
                bottlenecks['flops_bottleneck'] = "Lightning Indexer (O(L²) complexity)"
            elif self.flops_estimates['attention_ratio'] > 0.5:
                bottlenecks['flops_bottleneck'] = "Attention computation"
            else:
                bottlenecks['flops_bottleneck'] = "Other components"
        
        return bottlenecks

def profile_model(model, batch, seq_len: int, d_model: int, n_heads: int, 
                 save_report: str = None) -> ComputeProfiler:
    """Convenience function to profile a model"""
    profiler = ComputeProfiler()
    
    # Profile the model
    profiler.profile_sparse_attention(model, batch, seq_len, d_model, n_heads)
    
    # Print report
    profiler.print_report(save_report)
    
    # Identify bottlenecks
    bottlenecks = profiler.identify_bottlenecks()
    if bottlenecks:
        print("\n🎯 Identified Bottlenecks:")
        for key, value in bottlenecks.items():
            print(f"  {key}: {value}")
    
    return profiler

if __name__ == "__main__":
    # Example usage
    print("Compute Profiler - Example Usage")
    print("=" * 40)
    print("Use this profiler to identify computational bottlenecks:")
    print("1. Lightning Indexer overhead")
    print("2. Attention computation")
    print("3. Memory usage patterns")
    print("4. FLOPs distribution")
    print("\nExample:")
    print("profiler = ComputeProfiler()")
    print("profiler.profile_sparse_attention(model, batch, seq_len, d_model, n_heads)")
    print("profiler.print_report('profile_report.json')")
