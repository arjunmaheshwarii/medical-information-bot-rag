"""
Prompt templates for medical question-answering tasks.
Provides templates for consistent prompt formatting.
"""

from typing import Dict


class MedicalPromptTemplates:
    """Collection of prompt templates for medical RAG."""
    
    @staticmethod
    def medical_qa_template() -> str:
        """
        Template for medical question-answering.
        Uses context from retrieved documents.
        
        Returns:
            str: Prompt template with {context} and {question} placeholders
        """
        return """You are a medical information assistant. Your role is to answer questions using ONLY the medical information provided in the context below.

Instructions:
- Answer based ONLY on the context provided
- If the context doesn't contain relevant information, say so clearly
- Be accurate and cite specific information from the context
- Use simple, clear language suitable for patients

Context:
{context}

Question:
{question}

Answer:"""
    
    @staticmethod
    def medical_summary_template() -> str:
        """
        Template for medical text summarization.
        
        Returns:
            str: Prompt template for summarization
        """
        return """Summarize the following medical information concisely and clearly:

{context}

Summary:"""
    
    @staticmethod
    def medical_explanation_template() -> str:
        """
        Template for explaining medical concepts.
        
        Returns:
            str: Prompt template for explanation
        """
        return """Explain the following medical concept in simple terms based on the provided information:

Medical Information:
{context}

Concept to Explain:
{question}

Explanation:"""
    
    @staticmethod
    def get_template(template_name: str = "qa") -> str:
        """
        Get prompt template by name.
        
        Args:
            template_name (str): Name of template
                               - 'qa' (default): Question-answering
                               - 'summary': Summarization
                               - 'explain': Concept explanation
        
        Returns:
            str: Prompt template string
        """
        templates = {
            "qa": MedicalPromptTemplates.medical_qa_template(),
            "summary": MedicalPromptTemplates.medical_summary_template(),
            "explain": MedicalPromptTemplates.medical_explanation_template(),
        }
        return templates.get(template_name, templates["qa"])
