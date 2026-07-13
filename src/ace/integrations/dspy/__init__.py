"""ace.integrations.dspy — the ACE optimizer as a DSPy teleprompter.

One integration of the framework-agnostic ACE engine; prototype home for what
would become ``dspy.teleprompt.ace``.
"""

from ace.integrations.dspy.ace import ACE
from ace.integrations.dspy.adapter import DspyAdapter

__all__ = ["ACE", "DspyAdapter"]
