"""
FLAN-T5 language model wrapper for medical question-answering.
Provides simple interface for text generation.
"""

from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.utils.config import LLM_MODEL_NAME, LLM_MAX_TOKENS, LLM_TEMPERATURE
from .prompt_templates import MedicalPromptTemplates


class FlanT5Model:
    """
    FLAN-T5 model wrapper for question-answering and text generation.
    
    Features:
    - Automatic GPU/CPU detection
    - Batch inference for efficiency
    - Configurable generation parameters
    - Medical prompt templates
    
    Model: google/flan-t5-base (220M parameters, efficient)
    """
    
    def __init__(
        self,
        model_name: str = LLM_MODEL_NAME,
        max_tokens: int = LLM_MAX_TOKENS,
        temperature: float = LLM_TEMPERATURE,
    ):
        """
        Initialize FLAN-T5 model.
        
        Args:
            model_name (str): HuggingFace model identifier
                             Default: 'google/flan-t5-base'
            max_tokens (int): Maximum tokens to generate
                             Default: 200
            temperature (float): Sampling temperature (0.0-1.0)
                                Higher = more creative
                                Default: 0.7
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Auto-detect device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading {model_name} on {self.device}...")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
        # Set to evaluation mode
        self.model.eval()
        
        print(f"[OK] Model loaded successfully on {self.device}")
    
    def generate(
        self,
        context: str,
        question: str,
        template_name: str = "qa",
        max_tokens: int = None,
    ) -> str:
        """
        Generate answer to a question given context.
        
        Args:
            context (str): Retrieved document context
            question (str): User question
            template_name (str): Prompt template to use
                                ('qa', 'summary', 'explain')
            max_tokens (int): Max tokens to generate.
                             If None, uses default
        
        Returns:
            str: Generated answer
        
        Example:
            >>> model = FlanT5Model()
            >>> answer = model.generate(
            ...     context="Diabetes is...",
            ...     question="What is diabetes?"
            ... )
            >>> print(answer)
        """
        if max_tokens is None:
            max_tokens = self.max_tokens
        
        # Build prompt
        template = MedicalPromptTemplates.get_template(template_name)
        prompt = template.format(context=context, question=question)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        # Generate with no_grad for efficiency
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.95,
                top_k=50,
            )
        
        # Decode
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return answer.strip()
    
    def generate_batch(
        self,
        contexts: List[str],
        questions: List[str],
        max_tokens: int = None,
    ) -> List[str]:
        """
        Generate answers for multiple question-context pairs.
        
        Args:
            contexts (List[str]): List of contexts
            questions (List[str]): List of questions
            max_tokens (int): Max tokens per generation
        
        Returns:
            List[str]: List of generated answers
        
        Example:
            >>> answers = model.generate_batch(
            ...     contexts=[ctx1, ctx2],
            ...     questions=[q1, q2]
            ... )
        """
        if len(contexts) != len(questions):
            raise ValueError("contexts and questions must have same length")
        
        if max_tokens is None:
            max_tokens = self.max_tokens
        
        answers = []
        for context, question in zip(contexts, questions):
            answer = self.generate(context, question, max_tokens=max_tokens)
            answers.append(answer)
        
        return answers
    
    def __repr__(self) -> str:
        return f"FlanT5Model(model='{self.model_name}', device='{self.device}')"
