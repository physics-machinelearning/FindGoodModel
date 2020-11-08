from setuptools import setup

setup(
    name="auto_learning",
    version="0.0.1",
    description="",
    author="physics-machinelearning",
    packages=["auto_learning"],
    zip_safe=False,
    install_requires=["numpy", 'scikit-learn', "bayesian-optimization", "matplotlib"]
)