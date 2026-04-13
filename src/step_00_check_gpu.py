"""Step 0: Verify GPU and dependencies."""
import torch, sys

print("="*50+"\n  FINREASON — GPU CHECK\n"+"="*50)
if not torch.cuda.is_available():
    print("  ✗ No CUDA GPU found."); sys.exit(1)
print(f"  GPU:      {torch.cuda.get_device_name(0)}")
print(f"  VRAM:     {torch.cuda.get_device_properties(0).total_mem/1024**3:.1f} GB")
print(f"  PyTorch:  {torch.__version__}")
print(f"  CUDA:     {torch.version.cuda}")
x = torch.randn(1000,1000,device="cuda"); _ = x @ x.T; del x,_; torch.cuda.empty_cache()
print(f"  Compute:  ✓")
for p in ["transformers","trl","peft","bitsandbytes","accelerate","datasets","streamlit"]:
    try: m=__import__(p); print(f"  ✓ {p}: {getattr(m,'__version__','ok')}")
    except ImportError: print(f"  ✗ {p}: MISSING")
try: import unsloth; print(f"  ✓ unsloth: {unsloth.__version__}")
except ImportError: print(f"  - unsloth: not installed (optional)")
print("="*50)
