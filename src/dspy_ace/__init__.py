"""dspy_ace — the ACE optimizer as a DSPy teleprompter.

Prototype home for what would become ``dspy.teleprompt.ace``.
"""

from dspy_ace.ace import ACE
from dspy_ace.adapter import DspyAdapter

__all__ = ["ACE", "DspyAdapter"]
