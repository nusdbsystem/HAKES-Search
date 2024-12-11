from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name="hakes_client",
    version="0.1.0",
    author="guoyu",
    description="hakes client",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/hugy718/HAKES",
    packages=find_packages(),
    install_requires=[
        "requests==2.31.0",
        "numpy>=1.19.5"
    ],
)
