"""
Setup script for 3D CAD Geometry Analyzer
"""
from setuptools import setup, find_packages

setup(
    name="3d-cad-geometry-analyzer",
    version="0.1.0",
    description="A comprehensive tool for analyzing 3D CAD geometry",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "open3d>=0.15.1",
        "scipy>=1.7.0",
        "pyyaml>=6.0",
        "torch>=1.9.0",
        "torch-geometric>=2.0.0",
        "tensorflowjs>=3.18.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
