"""LLM module for text generation."""

from .flan_t5_model import FlanT5Model
from .prompt_templates import MedicalPromptTemplates

__all__ = ["FlanT5Model", "MedicalPromptTemplates"]
