"""``setup.py`` — metadata-only shim for tools that still require it.

The canonical metadata is in ``pyproject.toml``. This file is a
shim that delegates to ``setuptools`` so that ``pip install`` (which
in some flows still inspects ``setup.py``) works correctly. The
package's install requirements, optional extras, and console-script
entry points are all declared in ``pyproject.toml``.
"""

from __future__ import annotations

from setuptools import setup

setup()
