#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DRIMAPS: Deadlock-Resilient Intelligent Multi-Agent Path Finding System
Setup configuration script
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="drimaps",
    version="1.0.0",
    author="Ubaid Mushtaq Mir",
    author_email="ubaidmushtaq0786@gmail.com",
    description="Runtime deadlock resolution framework for Multi-Agent Path Finding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ubaid0786/DRIMAPS",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
        "paper": [
            "scipy>=1.10.0",
            "pandas>=1.5.0",
            "seaborn>=0.12.0",
        ],
    },
)
