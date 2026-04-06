"""
Step 0: Verify GPU, CUDA, and dependencies.
Run this FIRST before anything else.
"""
import torch
import sys

print("=" * 55)
print("  FINREASON — GPU & DEPENDENCY CHECK")
print("=" * 55)

# --- GPU ---
if not torch.cuda.is_available():
    print("\n  ✗ CUDA is NOT available.")
    print("    Install CUDA toolkit or use Google Colab.")
    sys.exit(1)

gpu = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_mem / 1024**3
print(f"\n  GPU:       {gpu}")
print(f"  VRAM:      {vram:.1f} GB")
print(f"  PyTorch:   {torch.__version__}")
print(f"  CUDA:      {torch.version.cuda}")

# Quick compute test
x = torch.randn(1000, 1000, device="cuda")
_ = x @ x.T
print(f"  Compute:   ✓ passed")
del x, _
torch.cuda.empty_cache()

# --- Key packages ---
packages = {
    "transformers": None,
    "trl": None,
    "peft": None,
    "bitsandbytes": None,
    "accelerate": None,
    "datasets": None,
    "streamlit": None,
}

for pkg in packages:
    try:
        mod = __import__(pkg)
        packages[pkg] = getattr(mod, "__version__", "installed")
    except ImportError:
        packages[pkg] = "MISSING"

print("\n  Dependencies:")
for pkg, ver in packages.items():
    mark = "✓" if ver != "MISSING" else "✗"
    print(f"    {mark} {pkg}: {ver}")

# --- Unsloth (optional) ---
try:
    import unsloth
    print(f"    ✓ unsloth: {unsloth.__version__} (recommended)")
except ImportError:
    print(f"    - unsloth: not installed (optional, saves ~30% VRAM)")

missing = [k for k, v in packages.items() if v == "MISSING"]
if missing:
    print(f"\n  ✗ Missing packages: {', '.join(missing)}")
    print(f"    Run: pip install {' '.join(missing)}")
    sys.exit(1)

print(f"\n  ✓ All checks passed. Ready to go.")
print("=" * 55)
