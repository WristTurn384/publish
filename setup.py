import os

import setuptools

with open("version.txt") as f:
    VERSION = f.read().strip()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="publish",
    version=VERSION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "torch",
    ],
    packages=['publish']
)

