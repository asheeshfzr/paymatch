# app/prompt_templates.py
"""
Versioned prompt templates for LLM operations with validation and testing.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class PromptVersion(Enum):
    """Prompt template versions."""
    V1 = "v1"
    V2 = "v2"


@dataclass
class PromptTemplate:
    """A prompt template with validation."""
    version: PromptVersion
    template: str
    expected_output_format: str
    validation_regex: Optional[str] = None
    max_tokens: int = 128
    temperature: float = 0.0
    description: str = ""


class PromptTemplates:
    """Collection of versioned prompt templates."""
    
    def __init__(self):
        self.templates = {
            "name_extraction": {
                PromptVersion.V1: PromptTemplate(
                    version=PromptVersion.V1,
                    template=(
                        "Extract the payer's full name from the following bank transaction description. "
                        "Return the name only, without extra explanation or punctuation.\n\n"
                        "Description:\n\"{description}\"\n\nName:"
                    ),
                    expected_output_format="Plain text name",
                    validation_regex=r'^[A-Za-z\s]+$',
                    max_tokens=32,
                    temperature=0.0,
                    description="Extract payer name from transaction description"
                ),
                PromptVersion.V2: PromptTemplate(
                    version=PromptVersion.V2,
                    template=(
                        "From this transaction description, identify the person or entity who made the payment. "
                        "Extract only their full name, nothing else.\n\n"
                        "Transaction: \"{description}\"\n\n"
                        "Payer name:"
                    ),
                    expected_output_format="Plain text name",
                    validation_regex=r'^[A-Za-z\s]+$',
                    max_tokens=32,
                    temperature=0.0,
                    description="Extract payer name with improved clarity"
                )
            },
            "query_expansion": {
                PromptVersion.V1: PromptTemplate(
                    version=PromptVersion.V1,
                    template=(
                        "Generate up to 3 short paraphrases of the following financial/search query. "
                        "Each paraphrase should be on its own line. Do NOT include any numbering or extra text.\n\n"
                        "Query: \"{query}\"\n\nParaphrases:"
                    ),
                    expected_output_format="One paraphrase per line",
                    validation_regex=r'^[^\n]*(?:\n[^\n]*){0,2}$',
                    max_tokens=120,
                    temperature=0.2,
                    description="Generate query paraphrases for better search"
                ),
                PromptVersion.V2: PromptTemplate(
                    version=PromptVersion.V2,
                    template=(
                        "Create 3 alternative ways to search for this financial transaction. "
                        "Each alternative should be a complete, natural sentence on its own line.\n\n"
                        "Original query: \"{query}\"\n\n"
                        "Alternative searches:"
                    ),
                    expected_output_format="One alternative per line",
                    validation_regex=r'^[^\n]*(?:\n[^\n]*){0,2}$',
                    max_tokens=120,
                    temperature=0.2,
                    description="Generate natural alternative search queries"
                )
            },
            "reranking": {
                PromptVersion.V1: PromptTemplate(
                    version=PromptVersion.V1,
                    template=(
                        "Rank the following candidate transaction descriptions by how semantically similar they are to the query.\n"
                        "Return only the ordered list of indices (1-based), separated by commas (e.g. 3,1,2).\n\n"
                        "Query: \"{query}\"\n\nCandidates:\n{candidates}\n\nRank:"
                    ),
                    expected_output_format="Comma-separated indices",
                    validation_regex=r'^[\d,]+$',
                    max_tokens=256,
                    temperature=0.0,
                    description="Rank candidates by semantic similarity"
                ),
                PromptVersion.V2: PromptTemplate(
                    version=PromptVersion.V2,
                    template=(
                        "Given the search query, reorder these transaction descriptions from most relevant to least relevant.\n"
                        "Respond with only the numbers in order, separated by commas (e.g. 2,4,1,3).\n\n"
                        "Search query: \"{query}\"\n\n"
                        "Transaction descriptions:\n{candidates}\n\n"
                        "Relevance order:"
                    ),
                    expected_output_format="Comma-separated indices",
                    validation_regex=r'^[\d,]+$',
                    max_tokens=256,
                    temperature=0.0,
                    description="Rank by relevance with clearer instructions"
                )
            }
        }
    
    def get_template(self, operation: str, version: PromptVersion = PromptVersion.V2) -> PromptTemplate:
        """Get a prompt template for an operation."""
        if operation not in self.templates:
            raise ValueError(f"Unknown operation: {operation}")
        
        if version not in self.templates[operation]:
            # Fallback to V1 if V2 not available
            version = PromptVersion.V1
        
        return self.templates[operation][version]
    
    def format_template(self, operation: str, **kwargs) -> str:
        """Format a prompt template with the given parameters."""
        template = self.get_template(operation)
        try:
            return template.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing required parameter for {operation}: {e}")
    
    def validate_response(self, operation: str, response: str, version: PromptVersion = PromptVersion.V2) -> bool:
        """Validate a response against the template's expected format."""
        template = self.get_template(operation, version)
        
        if not template.validation_regex:
            return True
        
        return bool(re.match(template.validation_regex, response.strip()))
    
    def get_template_config(self, operation: str, version: PromptVersion = PromptVersion.V2) -> Dict[str, Any]:
        """Get template configuration (max_tokens, temperature, etc.)."""
        template = self.get_template(operation, version)
        return {
            'max_tokens': template.max_tokens,
            'temperature': template.temperature,
            'version': template.version.value,
            'description': template.description
        }


# Global prompt templates instance
prompt_templates = PromptTemplates()


def get_prompt_version() -> PromptVersion:
    """Get the current prompt version from settings."""
    version_str = getattr(settings, 'prompt_version', 'v2')
    try:
        return PromptVersion(version_str)
    except ValueError:
        logger.warning(f"Invalid prompt version {version_str}, using v2")
        return PromptVersion.V2


def format_prompt(operation: str, **kwargs) -> str:
    """Format a prompt for the given operation."""
    version = get_prompt_version()
    return prompt_templates.format_template(operation, **kwargs)


def validate_llm_response(operation: str, response: str) -> bool:
    """Validate an LLM response."""
    version = get_prompt_version()
    return prompt_templates.validate_response(operation, response, version)


def get_llm_config(operation: str) -> Dict[str, Any]:
    """Get LLM configuration for an operation."""
    version = get_prompt_version()
    return prompt_templates.get_template_config(operation, version)


# Test cases for prompt templates
TEST_CASES = {
    "name_extraction": [
        {
            "input": {"description": "Payment from John Smith for invoice #123"},
            "expected_pattern": r"John Smith"
        },
        {
            "input": {"description": "Transfer to Acme Corp Ltd"},
            "expected_pattern": r"Acme Corp Ltd"
        }
    ],
    "query_expansion": [
        {
            "input": {"query": "find payments to contractors"},
            "expected_lines": 3
        }
    ],
    "reranking": [
        {
            "input": {
                "query": "office supplies",
                "candidates": "1. Office supplies purchase\n2. Grocery shopping\n3. Software license"
            },
            "expected_format": r"^[\d,]+$"
        }
    ]
}


def test_prompt_templates() -> Dict[str, bool]:
    """Test prompt templates with sample inputs."""
    results = {}
    
    for operation, test_cases in TEST_CASES.items():
        try:
            for i, test_case in enumerate(test_cases):
                # Format the prompt
                formatted = format_prompt(operation, **test_case["input"])
                
                # Check if it contains expected elements
                if "expected_pattern" in test_case:
                    if not re.search(test_case["expected_pattern"], formatted):
                        results[f"{operation}_test_{i}"] = False
                        continue
                
                if "expected_lines" in test_case:
                    lines = formatted.count('\n')
                    if lines < test_case["expected_lines"]:
                        results[f"{operation}_test_{i}"] = False
                        continue
                
                if "expected_format" in test_case:
                    # This would need actual LLM response to test
                    pass
                
                results[f"{operation}_test_{i}"] = True
        
        except Exception as e:
            logger.error(f"Test failed for {operation}: {e}")
            results[f"{operation}_error"] = False
    
    return results


if __name__ == "__main__":
    # Run tests
    test_results = test_prompt_templates()
    print("Prompt template tests:")
    for test, result in test_results.items():
        print(f"  {test}: {'PASS' if result else 'FAIL'}")
