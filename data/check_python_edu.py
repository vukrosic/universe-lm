from datasets import load_dataset

# Check if Python-Edu has content field or needs different loading
print("Loading Python-Edu with more details...")
ds_python = load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split="train", streaming=True)
iterator = iter(ds_python)

# Get first 3 examples to understand the pattern
for i in range(3):
    print(f"\n{'='*60}")
    print(f"Example {i+1}:")
    example = next(iterator)
    for key, value in example.items():
        print(f"{key}: {value}")

# Try loading in non-streaming mode to see full schema
print("\n" + "="*60)
print("Loading non-streaming to check features...")
try:
    ds_python_full = load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split="train[:5]")
    print(f"\nDataset features: {ds_python_full.features}")
    print(f"\nFirst example (full):")
    print(ds_python_full[0])
except Exception as e:
    print(f"Error loading non-streaming: {e}")
