"""Deprecated import path. Use ``ace.integrations.dspy`` instead.

Kept as a thin shim so existing ``from dspy_ace import ACE`` code keeps working.
"""

from ace.integrations.dspy import ACE, DspyAdapter
from ace.integrations.dspy.generator import ACEGenerator

__all__ = ["ACE", "DspyAdapter", "ACEGenerator"]
