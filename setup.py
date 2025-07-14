# setup.py
from setuptools import setup, find_packages

setup(
    name="EasySP",
    version="0.1.0", # please read feishu doc for version naming rule 
    packages=find_packages(),
    install_requires=[],
    author="1B1B team",
    author_email="your@email.com",
    description="A lightweight library for easy sequence parallelism",
    python_requires=">=3.7",
)
