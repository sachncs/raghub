from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup


BASE_DIR = Path(__file__).parent


def read_readme() -> str:
    return (BASE_DIR / "README.md").read_text(encoding="utf-8")


setup(
    name="retrieval-augmented-generation",
    version="1.0.0",
    description="Production-grade Dynamic RAG framework",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["docs", "tests", "config", "data"]),
    include_package_data=True,
    install_requires=[
        "pydantic>=2.8",
        "pypdf>=5.0",
        "numpy>=1.26",
        "PyYAML>=6.0",
    ],
    extras_require={
        "api": ["fastapi>=0.115", "uvicorn>=0.30", "python-multipart>=0.0.9"],
        "ui": ["streamlit>=1.37"],
        "zvec": ["zvec>=0.5.0"],
        "dev": ["pytest>=8.0", "reportlab>=4.2", "ruff>=0.6", "mypy>=1.10"],
    },
    python_requires=">=3.12",
)
