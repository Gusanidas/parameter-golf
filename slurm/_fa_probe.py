import importlib
for name in ("flash_attn", "flash_attn_interface", "flash_attn_2_cuda", "flash_attn_3_cuda"):
    try:
        m = importlib.import_module(name)
        v = getattr(m, "__version__", "?")
        f = getattr(m, "__file__", "?")
        print(f"OK   {name:24s} version={v} file={f}")
    except Exception as e:
        print(f"MISS {name:24s} {type(e).__name__}: {e}")

import torch
print(f"torch={torch.__version__}  cuda={torch.version.cuda}")
print(f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

# Confirm signatures
try:
    from flash_attn_interface import flash_attn_func as fa3
    import inspect
    print(f"FA3 signature: {inspect.signature(fa3)}")
except Exception as e:
    print(f"FA3 sig: {e}")
try:
    from flash_attn import flash_attn_func as fa2
    import inspect
    print(f"FA2 signature: {inspect.signature(fa2)}")
except Exception as e:
    print(f"FA2 sig: {e}")
