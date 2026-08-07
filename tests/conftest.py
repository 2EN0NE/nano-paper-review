"""
Shared fixtures and mock injection for tests that don't have onnxruntime installed.
"""

from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _inject_mock_modules():
    """Inject mock onnxruntime and tokenizers into sys.modules.

    When the system Python lacks onnxruntime / tokenizers (e.g. Python 3.9),
    tests that use ``patch("onnxruntime.InferenceSession")`` or
    ``patch("tokenizers.Tokenizer.from_file")`` would fail at fixture setup
    because the target modules aren't importable.

    This autouse fixture registers lightweight mocks in sys.modules before
    any test runs, so ``patch()`` can resolve its targets.  Individual
    tests/fixtures then replace the mock with their own controlled version.
    """
    import importlib
    import sys

    mods_added = []
    for name in ("onnxruntime", "tokenizers"):
        try:
            importlib.import_module(name)
        except ImportError:
            sys.modules[name] = MagicMock()
            mods_added.append(name)

    yield

    for m in mods_added:
        del sys.modules[m]
