from __future__ import annotations

import os
import re

from setuptools import setup


def get_version():
    pkg = "render_molecules"
    version_file = os.path.join(os.path.dirname(__file__), pkg, "__init__.py")

    try:
        with open(version_file) as file:
            verstrline = file.read()
    except OSError:
        pass
    else:
        version_re = r"^__version__ = ['\"]([^'\"]*)['\"]"
        mo = re.search(version_re, verstrline, re.M)
        if mo:
            return mo.group(1)
        else:
            print(f"Unable to find version in {version_file}")
            raise RuntimeError(
                f"If {version_file}.py exists, it is required to be well-formed"
            )


if __name__ == "__main__":
    setup(version=get_version())
