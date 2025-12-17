from datasets import load_dataset

# Check Python-Edu dataset structure
print("Loading Python-Edu dataset...")
ds_python = load_dataset("HuggingFaceTB/smollm-corpus", "python-edu", split="train", streaming=True)
iterator = iter(ds_python)

# Get first example
print("\nFirst example from Python-Edu:")
first_example = next(iterator)
print(f"Keys: {list(first_example.keys())}")
print(f"\nFirst example:")
for key, value in first_example.items():
    if isinstance(value, str) and len(value) > 200:
        print(f"{key}: {value[:200]}...")
    else:
        print(f"{key}: {value}")

# Check FineWeb and Cosmopedia too
print("\n" + "="*50)
print("Checking FineWeb-Edu...")
ds_fineweb = load_dataset("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup", split="train", streaming=True)
fineweb_example = next(iter(ds_fineweb))
print(f"Keys: {list(fineweb_example.keys())}")

print("\n" + "="*50)
print("Checking Cosmopedia...")
ds_cosmo = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True)
cosmo_example = next(iter(ds_cosmo))
print(f"Keys: {list(cosmo_example.keys())}")
