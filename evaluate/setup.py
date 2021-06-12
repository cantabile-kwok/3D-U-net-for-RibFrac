import shutil
from setuptools import find_packages, setup


NAME = "ribfrac"

with open("requirements.txt", encoding="utf-8") as f:
    REQUIRED = f.read().split("\n")

# required Python version
REQUIRES_PYTHON = ">=3.5.0"

setup(
    name=NAME,
    version="1.0.0",
    author="Jiancheng Yang, Kaiming Kuang",
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(),
    install_requires=REQUIRED
)

# cleanup
build_folders = [
    "dist",
    "build",
    "ribfrac.egg-info"
]
for folder in build_folders:
    shutil.rmtree(folder)
