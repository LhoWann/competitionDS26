"""Command-line entry points: `python -m cli.prepare | cli.train | cli.predict` (with `src` on PYTHONPATH)."""

import argparse


def config_parser(doc: str) -> argparse.ArgumentParser:
    """Parser with the shared `--config file.yaml` and `--set section.key=value ...` options."""
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", nargs="*", default=[])
    return parser