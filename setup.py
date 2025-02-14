from setuptools import setup, find_packages
import sys

# Check if the "benchmark" extra is being installed
installing_benchmark = any(arg.startswith("acfx[benchmark]") for arg in sys.argv)

# Default: Exclude `acfx.model`
excluded_packages = ["acfx.model"]

# Include `acfx.model` only when installing with `benchmark`
if installing_benchmark:
    excluded_packages = []

setup(
    name="acfx",
    version="0.1.0",
    packages=find_packages(exclude=excluded_packages),
)
