"""Setup configuration for claude-context-manager."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="claude-context-manager",
    version="0.4.0",
    author="Suzano AI",
    description="Production-ready conversation management for Claude API with automatic token tracking and context trimming.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/suzano-ai/claude-context-manager",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tiktoken>=0.5.0",
    ],
    include_package_data=True,
)
