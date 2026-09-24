from setuptools import setup, find_packages

setup(
    name="autoquantum",
    version="0.10.0",
    description="分子反应动力学全维量子动力学自动实现平台",
    author="AutoQuantum Team",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "scipy>=1.7.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "autoquantum=autoquantum.cli:main",
        ],
    },
)
