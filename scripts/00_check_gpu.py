"""Step 0: Verify GPU and dependencies before anything else."""
import torch, sys

print("=" * 50)
print("GPU & DEPENDENCY CHECK")
print("=" * 50)

if not torch.cuda.is_available():
    print("ERROR: No CUDA GPU found. Use Google Colab or install CUDA.")
    sys.exit(1)

print(f"GPU:        {torch.cuda.get_device_name(0)}")
print(f"VRAM:       {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
print(f"PyTorch:    {torch.__version__}")
print(f"CUDA:       {torch.version.cuda}")

x = torch.randn(1000, 1000, device="cuda")
_ = x @ x.T
print(f"Compute:    PASSED")
del x, _; torch.cuda.empty_cache()

for lib, name in [("unsloth", "Unsloth"), ("transformers", "Transformers"),
                   ("trl", "TRL"), ("peft", "PEFT"), ("bitsandbytes", "bitsandbytes"),
                   ("datasets", "Datasets"), ("streamlit", "Streamlit")]:
    try:
        mod = __import__(lib)
        ver = getattr(mod, "__version__", "ok")
        print(f"{name:15s} {ver} ✓")
    except ImportError:
        print(f"{name:15s} NOT INSTALLED ✗")

print("=" * 50)
