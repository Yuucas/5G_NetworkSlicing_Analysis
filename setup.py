from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="5g-network-slicing",
    version="0.1.0",
    author="AI Engineering Team",
    author_email="ai-team@example.com",
    description="AI-Driven 5G Network Slicing and Dynamic Resource Allocation System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourorg/5g-network-slicing",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Telecommunications Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "pre-commit>=3.4.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.2.0",
        ],
        "viz": [
            "streamlit>=1.27.0",
            "plotly>=5.17.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "5g-train=scripts.train_rl_agent:main",
            "5g-serve=src.api.app:main",
            "5g-optimize=scripts.optimize_resources:main",
        ],
    },
)
