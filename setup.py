import os

from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="holoformer",
    version="0.0.1",
    author="Adam Wentz",
    author_email="adam@adamwentz.com",
    description="Transformer using Holographic Reduced Representation for token mixing.",
    long_description=read("README.md"),
    license="MIT",
    url="https://github.com/awentzonline/holoformer",
    packages=find_packages(),
    install_requires=[
        'datasets',
        'numpy',
        'torch',
        'transformers',
        'pytorch_lightning',
    ]
)
