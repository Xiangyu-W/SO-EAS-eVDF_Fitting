"""Tiny nbformat-free notebook writer: build .ipynb files from (kind, source) cells.

Usage:
    from notebook_builder import Notebook
    nb = Notebook()
    nb.md("# Title")
    nb.code("print(1)")
    nb.write('path/to/file.ipynb')
"""
import json
from pathlib import Path

KERNEL_SPEC = {"display_name": "Python 3", "language": "python", "name": "python3"}
LANGUAGE_INFO = {"name": "python", "version": "3.10"}


class Notebook:
    def __init__(self):
        self.cells = []

    def md(self, source):
        self.cells.append({"cell_type": "markdown", "metadata": {},
                           "source": source.strip("\n")})
        return self

    def code(self, source):
        self.cells.append({"cell_type": "code", "execution_count": None,
                           "metadata": {}, "outputs": [], "source": source.strip("\n")})
        return self

    def write(self, path):
        for index, cell in enumerate(self.cells):
            cell["id"] = f"cell-{index:03d}"
        notebook = {"cells": self.cells,
                    "metadata": {"kernelspec": KERNEL_SPEC, "language_info": LANGUAGE_INFO},
                    "nbformat": 4, "nbformat_minor": 5}
        Path(path).write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
        return Path(path)
