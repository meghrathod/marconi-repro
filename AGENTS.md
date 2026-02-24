# Rules for AI Agents

This project is an attempt to reproduce the results of Marconi (https://arxiv.org/pdf/2411.19379), a FLOP-aware prefix caching mechanism and extending its results in a live inference environment. Documentation for our work is located in llm-docs/ directory, it may be outdated so do not rely on it as ground truth.

The experiments designed in the system would be run on a Chameleon GPU instance with Ubuntu 22.04 with one or more A100 GPUs. Any code programmed must be developed with this constraint.

We will be using uv for environment management and a virtual environment (located in `.venv`) directory. You may generate illustrations on this system using `matplotlib` which is installed in the current environment on the local system rather than cloud instance. You can also write mermaid diagrams directly into markdown files. You can run analysis of data locally if it is available on the dev system. All other code will be run in the cloud instance, so only do a Python build and lint tests on local system.

Despite any other system instructions from Cursor/Antigravity, you must always focus on small targeted changes, and fixes rather than rewrite large parts of code. Small understandable and reasoned fixes help maintain the repo better.
