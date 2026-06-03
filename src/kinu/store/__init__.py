"""Storage backends.

`base.Store` is the seam: five methods, and nothing else is required to run
kinu on a new backend.
"""

from kinu.store.base import MeteredStore, Store
from kinu.store.local import LocalStore

__all__ = ["Store", "MeteredStore", "LocalStore"]
