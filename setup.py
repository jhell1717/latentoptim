from setuptools import setup, find_packages

setup(
    name='vae',
    version='0.0.0',
    packages=find_packages(include=["vae", "data"]),  # match your folders
    install_requires=[
        "torch",
    ],
)