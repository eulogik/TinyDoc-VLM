from setuptools import setup, find_packages

setup(
    name="tinydoc",
    version="0.1.3",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "tinydoc": ["schemas/*.json"],
    },
    install_requires=[
        "pydantic>=2.8.0",
        "pillow>=10.0.0",
        "jsonschema>=4.20.0",
    ],
    extras_require={
        "ocr": ["pytesseract>=0.3.10"],
        "ocr-pp": ["rapidocr-onnxruntime>=1.3.0"],
        "smolvlm2": [
            "torch>=2.2.0",
            "transformers>=4.48.0",
            "numpy>=1.26.0",
            "sentencepiece>=0.2.0",
        ],
        "onnx": ["onnxruntime>=1.19.0", "optimum>=1.22.0"],
        "all": [
            "pytesseract>=0.3.10",
            "rapidocr-onnxruntime>=1.3.0",
            "torch>=2.2.0",
            "transformers>=4.48.0",
            "numpy>=1.26.0",
            "sentencepiece>=0.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tinydoc=tinydoc.cli:main",
        ],
    },
    author="eulogik",
    author_email="hello@eulogik.com",
    description=(
        "Local-first grounded document extraction SDK — schema, evidence, "
        "confidence. Free Ollama/Qwen engine by default; no API key."
    ),
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
