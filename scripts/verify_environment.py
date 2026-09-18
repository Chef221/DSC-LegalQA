from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

"""Authoritative environment verification for DSC-LegalQA P70 reproduction.

Proven Historical Runtime Authority (p70_public1000_dual_t4_runner_r3.py:L110):
- transformers == 5.17.0
- peft == 0.20.0
- accelerate == 1.15.0
- bitsandbytes == 0.50.2
- torchao explicitly uninstalled / absent
- safetensors installed (required, but unpinned in historical runner)
- torch preinstalled by host (Kaggle Python 3.10+ / Modal; exact torch version was not pinned)

Verification Modes:
- default (informative/compatibility): logs installed versions, warns on mismatches, permits dev execution.
- --strict (certified reproduction): fails closed (exit code 1) on any mismatch with proven historical pins.
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_LOGGER = logging.getLogger("verify_environment")

# Exact proven historical production package versions
EXACT_PROVEN_PRODUCTION_PINS: dict[str, str] = {
    "transformers": "5.17.0",
    "peft": "0.20.0",
    "accelerate": "1.15.0",
    "bitsandbytes": "0.50.2",
}

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

    # 1. Check core packages exist
    for pkg in EXPECTED_CORE_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "unknown")
            _LOGGER.info(f"[PASS] {pkg} is installed (version: {version})")
        except ImportError:
            _LOGGER.error(f"[FAIL] Required package '{pkg}' is missing!")
            success = False

    # 2. Check disallowed packages (torchao must be strictly absent)
    for pkg in DISALLOWED_PACKAGES:
        try:
            importlib.import_module(pkg)
            msg = f"Disallowed package '{pkg}' is present in environment! Must be uninstalled to avoid PEFT conflict."
            if strict:
                _LOGGER.error(f"[FAIL-CLOSED] {msg}")
                success = False
            else:
                _LOGGER.warning(f"[WARNING] {msg}")
        except ImportError:
            _LOGGER.info(f"[PASS] Disallowed package '{pkg}' is absent as required.")

    # 3. Check exact versions against proven production pins
    for pkg, expected_ver in EXACT_PROVEN_PRODUCTION_PINS.items():
        try:
            mod = importlib.import_module(pkg)
            actual_ver = getattr(mod, "__version__", "unknown")
            if actual_ver == expected_ver:
                _LOGGER.info(f"[PASS] Exact proven pin match for {pkg}: {actual_ver}")
            else:
                msg = f"Version mismatch for {pkg}: installed={actual_ver}, proven production pin={expected_ver}"
                if strict:
                    _LOGGER.error(f"[FAIL-CLOSED] {msg}")
                    success = False
                else:
                    _LOGGER.warning(f"[WARNING] {msg}")
        except ImportError:
            if pkg == "bitsandbytes":
                _LOGGER.info("[INFO] bitsandbytes not found (optional in CPU mode; required for quantized GPU execution).")
                if strict:
                    _LOGGER.error(f"[FAIL-CLOSED] bitsandbytes=={expected_ver} required in strict production mode.")
                    success = False
            else:
                _LOGGER.error(f"[FAIL] Missing package {pkg} for pin check.")
                success = False

    return success


def check_torch_and_cuda() -> bool:
    try:
        import torch
        _LOGGER.info(f"PyTorch version: {torch.__version__} (observed runtime host)")
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


def check_qwen35_compatibility(strict: bool = False) -> bool:
    try:
        import transformers
        has_native_class = hasattr(transformers, "Qwen3_5ForConditionalGeneration")
        _LOGGER.info(f"Transformers Qwen3_5ForConditionalGeneration native class available: {has_native_class}")

        if not has_native_class:
            msg = (
                "Qwen3_5ForConditionalGeneration class is missing in installed transformers. "
                "Canonical reproduction lock requires transformers == 5.17.0."
            )
            if strict:
                _LOGGER.error(f"[FAIL-CLOSED] {msg}")
                return False
            else:
                _LOGGER.warning(f"[WARNING] {msg} Falling back to generic causal LM is permitted only in compatibility mode.")
                return True
        return True
    except Exception as e:
        _LOGGER.error(f"[FAIL] Qwen3.5 compatibility check failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify execution environment for DSC-LegalQA.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enforce exact proven production pins (transformers==5.17.0, peft==0.20.0, etc.) and native Qwen3.5 class.",
    )
    args = parser.parse_args()

    mode_label = "STRICT PRODUCTION LOCK" if args.strict else "INFORMATIVE / COMPATIBILITY"
    _LOGGER.info(f"=== DSC-LegalQA Reproduction Environment Verification [{mode_label}] ===")

    ok_py = check_python_version()
    ok_pkg = check_packages(strict=args.strict)
    ok_torch = check_torch_and_cuda()
    ok_qwen = check_qwen35_compatibility(strict=args.strict)

    if ok_py and ok_pkg and ok_torch and ok_qwen:
        _LOGGER.info(f"=== [SUCCESS] Environment verification PASSED ({mode_label}) ===")
        sys.exit(0)
    else:
        _LOGGER.error(f"=== [FAIL-CLOSED] Environment verification FAILED ({mode_label}) ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
