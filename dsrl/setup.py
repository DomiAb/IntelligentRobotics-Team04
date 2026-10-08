from setuptools import setup, find_packages

setup(
    name="dsrl",
    version="0.1.0",
    packages=find_packages() + ["dsrl"],
    package_dir={"dsrl": "."},
)