# Package root: makes `src/agentflow` importable as `agentflow` after
# `pip install -e .` (src layout from pyproject.toml).
# Interview angle: keep the public surface tiny here — version + docstring —
# and import heavy modules (graph, API) lazily from subpackages so
# `import agentflow` stays cheap in tests and CLIs.

"""Multi-agent research assistant with LangGraph routing, MCP tools, and FastAPI."""

# Keep in sync with [project].version in pyproject.toml.
# Trade-off vs importlib.metadata.version("agentflow"): a literal works
# before the package is installed; metadata is better once you ship wheels.
__version__ = "0.1.0"

__all__ = ["__version__"]
