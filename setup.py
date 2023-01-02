from xml.etree.ElementInclude import include
from setuptools import setup, find_packages

setup(
    python_requires='>=3.10.0',
    packages=find_packages(
        include=['Generation-of-categorical-synthetic-data.*']
        ),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'torch>=1.8.0,<2',
        'sdv',
        'seaborn',
        'matplotlib',
        'SDMetrics'
    ]
)
