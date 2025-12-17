from dataclasses import dataclass
from typing import Optional, Callable, Union
import logging

logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    # Dataset source (supports HF dataset path format)
    dataset_path: str = "HuggingFaceTB/smollm-corpus"
    dataset_name: Optional[str] = "cosmopedia-v2"
    split: str = "train"
    
    # Tokenizer
    tokenizer_name: str = "HuggingFaceTB/SmolLM2-135M"
    use_fast: bool = True
    trust_remote_code: bool = False
    
    # Sequence processing
    seq_length: int = 512
    # stride: removed (unused)
    
    # Limits
    num_samples: Optional[int] = None  # Limit number of documents
    
    # Columns and preprocessing
    text_column: str = "text"
    # preprocessing_fn: removed (unused)
    # filter_fn: removed (unused)
    
    # Caching
    cache_dir: Optional[str] = "./hf_cache"
    num_proc: Optional[int] = None  # Parallel processing for .map()
    
    # Streaming
    streaming: bool = True  # Stream dataset to avoid downloading everything upfront

    # Persistence
    # save_to_disk / load_from_disk: removed (unused/handled externally)

    def __post_init__(self) -> None:
        # Validate dataset_path
        if not self.dataset_path or not isinstance(self.dataset_path, str):
            raise ValueError("dataset_path must be a non-empty string")
        if not self.dataset_path.strip():
            raise ValueError("dataset_path cannot be empty or whitespace")
        
        # Validate tokenizer_name
        if not self.tokenizer_name or not isinstance(self.tokenizer_name, str):
            raise ValueError("tokenizer_name must be a non-empty string")
        if not self.tokenizer_name.strip():
            raise ValueError("tokenizer_name cannot be empty or whitespace")
        
        # Validate split
        if not self.split or not isinstance(self.split, str):
            raise ValueError("split must be a non-empty string")
        if not self.split.strip():
            raise ValueError("split cannot be empty or whitespace")
        
        # Validate seq_length
        if not isinstance(self.seq_length, int):
            raise TypeError(f"seq_length must be an integer, got {type(self.seq_length).__name__}")
        if self.seq_length <= 0:
            raise ValueError(f"seq_length must be positive, got {self.seq_length}")
        
        # Validate num_samples
        if self.num_samples is not None:
            if not isinstance(self.num_samples, int):
                raise TypeError(f"num_samples must be an integer, got {type(self.num_samples).__name__}")
            if self.num_samples <= 0:
                raise ValueError(f"num_samples must be positive, got {self.num_samples}")
        
        # Validate text_column
        if not self.text_column or not isinstance(self.text_column, str):
            raise ValueError("text_column must be a non-empty string")
        if not self.text_column.strip():
            raise ValueError("text_column cannot be empty or whitespace")
        
        # Validate num_proc
        if self.num_proc is not None:
            if not isinstance(self.num_proc, int):
                raise TypeError(f"num_proc must be an integer, got {type(self.num_proc).__name__}")
            if self.num_proc <= 0:
                raise ValueError(f"num_proc must be positive, got {self.num_proc}")
