# Contributing to TinyDoc-VLM

Thanks for your interest in contributing!

## Getting Started

1. Fork the repo
2. Create a branch: `git checkout -b my-feature`
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Make your changes
5. Run tests: `pytest tests/`
6. Submit a PR

## Development Setup

```bash
git clone https://github.com/your-username/TinyDoc-VLM.git
cd TinyDoc-VLM
pip install -e .
```

## Training Contributions

To add new training data formats or evaluation benchmarks:

1. Add your data loader in `tinydoc_vlm/data/`
2. Add eval script in `evaluation/`
3. Test with: `python training/fast_train.py --steps 100 --batch-size 1`

## Code Style

- Follow existing patterns in the codebase
- Run `ruff check .` before committing
- Keep PRs focused — one feature per PR

## Reporting Issues

Open an issue with:
- What you expected
- What happened
- Steps to reproduce
- Your environment (OS, Python version, GPU)

## License

By contributing, you agree your code will be licensed under Apache 2.0.
