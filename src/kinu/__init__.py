"""kinu — grind documents into a searchable index.

*kinu* is Swahili for mortar and pestle. You grind documents down; what's
left is searchable. Everything durable lives in an object store; the query
engine is stateless and reads it back with ranged reads.
"""

from kinu.engine import Engine

__version__ = "0.1.0"
__all__ = ["Engine", "__version__"]
