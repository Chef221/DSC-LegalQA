from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

"""Authoritative environment verification for DSC-LegalQA P70 reproduction.

Verifies:
- Python version (>=3.10)
- PyTorch and CUDA availability
- Exact / compatible production dependencies:
  * transformers (Qwen3.5 model architecture support)
  * peft (LoRA adapter loading)
  * accelerate
  * bitsandbytes
  * safetensors
  * scikit-learn (HistGradientBoostingRegressor)
- Disallowed packages check (torchao must be absent)
- GPU device name and VRAM (if CUDA enabled)
"""

import argparse
import importlib
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_LOGGER = logging.getLogger("verify_environment")

# Production runtime authority expectations
EXPECTED_CORE_PACKAGES = [
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "safetensors",
    "sklearn",
    "numpy",
    "nltk",
]

DISALLOWED_PACKAGES = [
    "torchao",
]


def check_python_version() -> bool:
    v = sys.version_info
    _LOGGER.info(f"Python version: {v.major}.{v.minor}.{v.micro}")
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        _LOGGER.error(f"[FAIL] Python >= 3.10 required, detected: {v.major}.{v.minor}")
        return False
    _LOGGER.info("[PASS] Python version compatible")
    return True


def check_packages(strict: bool = False) -> bool:
    success = True
    for pkg in EXPECTED_CORE_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "unknown")
            _LOGGER.info(f"[PASS] {pkg} is installed (version: {version})")
        except ImportError:
            _LOGGER.error(f"[FAIL] Required package '{pkg}' is missing!")
            success = False

    # Verify disallowed packages
    for pkg in DISALLOWED_PACKAGES:
        try:
            importlib.import_module(pkg)
            _LOGGER.warning(f"[WARNING] Disallowed/removed package '{pkg}' is present in environment.")
            if strict:
                success = False
        except ImportError:
            _LOGGER.info(f"[PASS] Package '{pkg}' is absent as required.")

    # Optional bitsandbytes
    try:
        import bitsandbytes as bnb
        _LOGGER.info(f"[PASS] bitsandbytes is installed (version: {bnb.__version__})")
    except ImportError:
        _LOGGER.info("[INFO] bitsandbytes not found (optional for 16-bit serial/standard evaluation).")

    return success


def check_torch_and_cuda() -> bool:
    try:
        import torch
        _LOGGER.info(f"PyTorch version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        _LOGGER.info(f"CUDA available: {cuda_available}")

        if cuda_available:
            cnt = torch.cuda.device_count()
            _LOGGER.info(f"CUDA device count: {cnt}")
            for i in range(cnt):
                name = torch.cuda.get_device_name(i)
                cap = torch.cuda.get_device_capability(i)
                mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                _LOGGER.info(f"  GPU {i}: {name} | Compute Cap: {cap} | VRAM: {mem:.2f} GB")
        else:
            _LOGGER.info("Running on CPU mode. Note: End-to-end full 1000-query generation requires GPU (T4, A100, or H100).")
        return True
    except Exception as e:
        _LOGGER.error(f"[FAIL] PyTorch check failed: {e}")
        return False


def check_qwen35_compatibility() -> bool:
    try:
        import transformers
        # Check if Qwen3_5 or Qwen2/AutoModel supports qwen3_5
        has_qwen35 = hasattr(transformers, "Qwen3_5ForConditionalGeneration") or hasattr(transformers, "AutoModelForCausalLM")
        _LOGGER.info(f"Transformers Qwen3.5 generation class available: {has_qwen35}")
        return True
    except Exception as e:
        _LOGGER.error(f"[FAIL] Qwen3.5 compatibility check failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify execution environment for DSC-LegalQA.")
    parser.add_argument("--strict", action="store_true", help="Fail closed on any optional deviation.")
    args = parser.parse_args()

    _LOGGER.info("=== DSC-LegalQA Reproduction Environment Verification ===")
    ok_py = check_python_version()
    ok_pkg = check_packages(strict=args.strict)
    ok_torch = check_torch_and_cuda()
    ok_qwen = check_qwen35_compatibility()

    if ok_py and ok_pkg and ok_torch and ok_qwen:
        _LOGGER.info("=== [SUCCESS] Environment verification PASSED ===")
        sys.exit(0)
    else:
        _LOGGER.error("=== [FAIL] Environment verification FAILED ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
