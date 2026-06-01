"""Pytest session bootstrap.

Import torch FIRST (before pandas/sklearn get pulled in by test collection). On Windows,
pandas/sklearn load OpenMP/VC runtime DLLs that break torch's later DLL initialization
(WinError 1114). Loading torch up front avoids the conflict. No-op where torch is absent or on
Linux/Colab, where import order does not matter.
"""
try:  # pragma: no cover - environment dependent
    import torch  # noqa: F401
except Exception:
    pass
