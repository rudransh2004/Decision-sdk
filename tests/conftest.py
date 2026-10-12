import os

# Accelerate otherwise trains on MPS on macOS, where SDPA rejects dropout, and
# float32 results then differ from the CPU runners in CI.
os.environ.setdefault("ACCELERATE_USE_CPU", "true")
