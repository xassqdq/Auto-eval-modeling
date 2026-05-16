# setup.py
"""
AutoEval-Modeling 安装脚本
用法:
    pip install -e .          # 开发模式安装
    pip install .             # 正式安装
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取 README
long_description = ""
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

# 读取依赖
requirements = []
req_path = Path(__file__).parent / "requirements.txt"
if req_path.exists():
    with open(req_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 跳过注释行和空行
            if line and not line.startswith("#"):
                requirements.append(line)

setup(
    name="auto-eval-modeling",
    version="0.1.0",
    author="AutoEval Team",
    author_email="autoeval@example.com",
    description="面向评价类数学建模的自动化工作流引擎与代码生成系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/auto-eval-modeling",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "llm": [
            "openai>=1.0.0",
            "transformers>=4.35.0",
        ],
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "auto-eval=src.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    package_data={
        "": [
            "templates/latex/**/*.tex",
            "templates/python/**/*.j2",
            "configs/*.yaml",
        ]
    },
)