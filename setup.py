"""
Setup script for Sanskrit Text Segmentation project.

Install in development mode:
    pip install -e .
"""

from setuptools import setup, find_packages

setup(
    name="sanskrit_text_segmentation",
    version="0.1.0",
    description=(
        "Deep learning pipeline for segmenting Sanskrit Devanagari text "
        "from decorative/non-text regions in scanned manuscripts."
    ),
    author="Sanskrit Segmentation Research Team",
    python_requires=">=3.11",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "torchmetrics>=1.2.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "albumentations>=1.3.1",
        "matplotlib>=3.8.0",
        "PyYAML>=6.0.1",
        "tensorboard>=2.15.0",
        "tqdm>=4.66.0",
        "numpy>=1.24.0",
        "scikit-image>=0.22.0",
    ],
    entry_points={
        "console_scripts": [
            "sanskrit-seg=src.main:main",
        ],
    },
)
