"""Setup script for TikTok Analytics"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="tiktok-analytics",
    version="1.0.0", 
    description="Production-grade TikTok simulation and manipulation detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "matplotlib>=3.4.0",
        "plotly>=5.0.0",
        "streamlit>=1.0.0",
        "networkx>=2.6.0",
        "scikit-learn>=1.0.0",
    ],
    entry_points={
        "console_scripts": [
            "tiktok-simulate=simulator.core_simulator:main",
            "tiktok-analyze=analysis.anomaly_detection:main", 
            "tiktok-dashboard=dashboard.web_dashboard:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research", 
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
