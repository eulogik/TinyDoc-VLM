from setuptools import setup, find_packages

setup(
    name="tinydoc",
    version="0.1.0",
    packages=find_packages(),
    install_packages=["pydantic", "pillow", "pdf2image", "torch", "numpy"],
    author="eulogik",
    description="Python SDK for TinyDoc-VLM document understanding",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires=">=3.9",
)
