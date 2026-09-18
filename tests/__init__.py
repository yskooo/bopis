"""BOPIS test package.

This file exists for a specific reason: without it, ``tests/`` is only a
*namespace* package candidate, and an unrelated regular package named ``tests``
installed in ``site-packages`` shadows it. That made the documented invocation

    python -m unittest tests.test_stats

fail with ``ModuleNotFoundError: No module named 'tests.test_stats'`` even though
the file is present. Declaring a regular package here puts the project's copy
first on ``sys.path`` (``python -m`` prepends the working directory), so the
per-module and discovery invocations both resolve to this directory.

Full suite:

    python -m unittest discover -s tests -t .
"""
