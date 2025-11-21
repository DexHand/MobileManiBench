"""Installation script for the 'isaacsim' python package."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    'decord',
    'ffmpeg-python',
    'transforms3d',
    'open3d',
    'trimesh',
    'pybullet',
    'pycollada',
    'opencv-python',
    'rsl-rl-lib==2.2.4',
    'numba==0.61.2',
]

# Installation operation
setup(
    name="unimanip",
    author="DexHand",
    version="1.0",
    description="Benchmark environments for Universal Manipulation in NVIDIA IsaacLab.",
    keywords=["robotics", "rl"],
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.10"],
    zip_safe=False,
)
