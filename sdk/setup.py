from setuptools import setup, find_packages

setup(
    name="tinydoc",
    version="0.1.2",
    packages=find_packages(),
    install_requires=[
        "pydantic>=2.8.0",
        "pillow>=10.0.0",
        "torch>=2.2.0",
        "numpy>=1.26.0",
        "transformers>=4.48.0",
        "sentencepiece>=0.2.0",
    ],
    extras_require={
        "onnx": ["onnxruntime>=1.19.0", "optimum>=1.22.0"],
    },
    author="eulogik",
    author_email="hello@eulogik.com",
    description="Python SDK for TinyDoc-VLM document understanding — the world's smallest document-specialist VLM",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/eulogik/TinyDoc-VLM",
    project_urls={
        "HuggingFace Model": "https://huggingface.co/eulogik/TinyDoc-VLM-256M",
        "HuggingFace Space": "https://huggingface.co/spaces/eulogik/TinyDoc-VLM",
        "Bug Tracker": "https://github.com/eulogik/TinyDoc-VLM/issues",
        "Documentation": "https://github.com/eulogik/TinyDoc-VLM#readme",
        "Website": "https://eulogik.github.io/TinyDoc-VLM/",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Developers",
    ],
    python_requires=">=3.9",
)
