import runpy, sys, os
# Load the analyzer as a module object without executing __main__, then force RUN_DIR.
import importlib.util
spec = importlib.util.spec_from_file_location("ckpt_norms", "/repo/analyze_ckpt_norms.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.RUN_DIR = "/ckpt"
mod.MAX_STEPS = 8
mod.main()
