"""BOPIS -- Bayesian Optimization and Pareto-Based Intelligent Configuration
Selection for Energy-Efficient Local LLM Inference.

This package implements the methodology of Chapter 3 of the BOPIS thesis.

Dependency policy
-----------------
Every module in this package EXCEPT ``bopis.quality.bertscore`` must import only
the Python standard library. That constraint is enforced mechanically by
``tests/test_stdlib_only.py``; do not add third-party imports to the core.

``bopis.quality.bertscore`` is the single declared *evaluation* dependency
(``transformers`` + ``torch``) and runs as a separate offline scoring stage so
that the measurement core never loads it.
"""

__version__ = "0.1.0"

# Deliberately no eager imports of submodules: `bopis.quality.bertscore` pulls in
# torch, and `python -m bopis profile` must stay fast and dependency-free.

__all__ = ["__version__"]
