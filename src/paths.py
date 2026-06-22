from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def p(*parts):
    return ROOT.joinpath(*parts)