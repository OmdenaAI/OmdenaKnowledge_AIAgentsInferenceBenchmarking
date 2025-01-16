from setuptools import setup, find_packages

setup(
    name="autogen_crop_yield",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "autogen",
        "numpy",
        "pandas",
        "psutil",
    ],
) 