from setuptools import setup, find_packages

requirements = []

setup(
    name="data_analysis",
    version="0.0.0",
    packages=find_packages(),
    long_description=open("README.rst").read(),
    install_requires=requirements,
    include_package_data=True,
)
