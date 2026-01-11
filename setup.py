"""
Setup script for sensitivity-compiler-testing package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip() 
        for line in requirements_path.read_text().splitlines() 
        if line.strip() and not line.startswith('#')
    ]

setup(
    name="sensitivity-compiler-testing",
    version="0.1.0",
    author="H. M. Shujaat Zaheer",
    author_email="shujabis@gmail.com",
    description="Sensitivity-theoretic framework for compiler testing using chaos theory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hmshujaatzaheer/sensitivity-compiler-testing",
    project_urls={
        "Bug Reports": "https://github.com/hmshujaatzaheer/sensitivity-compiler-testing/issues",
        "Source": "https://github.com/hmshujaatzaheer/sensitivity-compiler-testing",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Testing",
        "Topic :: Software Development :: Compilers",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "pyyaml>=5.4.0",
        "click>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "viz": [
            "matplotlib>=3.4.0",
            "rich>=12.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sct=sensitivity_testing.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
