from setuptools import setup, find_packages

setup(
    name="bricks-ml3",
    version="0.1.2",
    author="Irvin Umana",
    description="bricks-ml3 -- Three Environment MLOps Project Template for Databricks",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    # lightgbm and shap are pre-installed on Databricks ML Runtime.
    # Declaring them here causes pip to reinstall/upgrade them on the
    # cluster, which crashes the driver. Keep them in requirements.txt
    # for local development only.
    install_requires=[],
    python_requires=">=3.10",
)
