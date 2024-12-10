#! /usr/bin/env python3
"""Define several XHTML document strings to be used in SwordTest.

Unlike the original program written in Java, a large portion of the
XHTML code is defined separately here to be used as format strings."""

__author__ = 'Stephen Paul Chappell'
__date__ = '2 May 2019'
__version__ = '$Revision: 4 $'

import math
import pathlib
import sys


def __getattr__(name):
    """Get HTML templates from ROOT while caching the file contents."""
    if name.isupper():  # Is the name for a constant value?
        path = (__getattr__.ROOT / name.casefold()).with_suffix('.html')
        if path.is_file():
            modified = path.stat().st_mtime
            cache_time, text = __getattr__.CACHE.get(name, (-math.inf, None))
            if cache_time < modified:
                with path.open() as file:
                    text = file.read()
                __getattr__.CACHE[name] = modified, text
            return text
        else:
            __getattr__.CACHE.pop(name, None)
    raise AttributeError(name)


__getattr__.CACHE = {}
__getattr__.ROOT = pathlib.Path(sys.argv[0]).parent / 'templates'
