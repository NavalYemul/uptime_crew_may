"""
prompt_engineering.py — Prompt Engineering Techniques
======================================================
Covers: zero-shot, few-shot, chain-of-thought prompting,
        token counting with tiktoken, cost estimation,
        and structured extraction with Instructor.

Why prompt engineering?
  The quality of your prompt is the #1 factor in LLM output quality.
  These patterns (zero-shot, few-shot, CoT) are used in every production
  AI system — from simple chatbots to complex multi-agent pipelines.

Run: python -m day4.prompt_engineering
"""

from __future__ import annotations

from typing import Optional
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel


# ══════════════════════════════════════════════════════
# PROMPT TEMPLATES
# ══════════════════════════════════════════════════════

def zero_shot_prompt(task: str) -> PromptTemplate:
    """Return a zero-shot PromptTemplate for the given task.

    Zero-shot: provide only the task description, no examples.
    Works well for general-purpose tasks where the LLM has enough
    training data to handle the domain.

    Args:
        task: Description of the task (e.g. "Classify the sentiment of this review").

    Returns:
        PromptTemplate with a single {input} variable.
    """
    template = f"{task}\n\nInput: {{input}}\nOutput:"
    return PromptTemplate(input_variables=["input"], template=template)


def few_shot_prompt(task: str, examples: list[dict]) -> PromptTemplate:
    """Return a few-shot PromptTemplate with examples formatted into the prompt.

    Few-shot: provide 2-5 worked examples before the actual input.
    Dramatically improves output format consistency.

    Args:
        task:     Description of the task.
        examples: List of dicts with "input" and "output" keys.

    Returns:
        PromptTemplate with a single {input} variable.
    """
    example_text = "\n\n".join(
        f"Input: {ex['input']}\nOutput: {ex['output']}"
        for ex in examples
    )
    template = (
        f"{task}\n\n"
        f"Examples:\n{example_text}\n\n"
        f"Now apply the same pattern:\n"
        f"Input: {{input}}\nOutput:"
    )
    return PromptTemplate(input_variables=["input"], template=template)


def chain_of_thought_prompt(task: str) -> PromptTemplate:
    """Return a chain-of-thought PromptTemplate.

    Chain-of-Thought (CoT): append "Let's think step by step." to encourage
    the LLM to reason before producing the final answer. Shown to significantly
    improve accuracy on multi-step reasoning tasks.

    Args:
        task: Description of the task.

    Returns:
        PromptTemplate with a single {input} variable.
    """
    template = (
        f"{task}\n\n"
        f"Input: {{input}}\n\n"
        f"Let's think step by step.\n"
        f"Output:"
    )
    return PromptTemplate(input_variables=["input"], template=template)


# ══════════════════════════════════════════════════════
# TOKEN COUNTING
# ══════════════════════════════════════════════════════

def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in text using tiktoken.

    Args:
        text:  Input string to tokenise.
        model: Model name to select the correct BPE encoding.

    Returns:
        Number of tokens as an integer.
    """
    import tiktoken
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


# ══════════════════════════════════════════════════════
# COST ESTIMATION
# ══════════════════════════════════════════════════════

# Approximate input costs in USD per 1M tokens (as of 2024)
_COST_PER_MILLION: dict[str, float] = {
    "gpt-4o":        5.00,
    "gpt-4o-mini":   0.15,
    "gpt-4-turbo":  10.00,
    "gpt-3.5-turbo": 0.50,
    "claude-3-haiku": 0.25,
    "claude-3-sonnet": 3.00,
    "claude-3-opus": 15.00,
}


def estimate_cost(token_count: int, model: str = "gpt-4o-mini") -> dict:
    """Estimate the cost in USD for a given number of input tokens.

    Args:
        token_count: Number of tokens to price.
        model:       Model identifier to look up cost.

    Returns:
        Dict with input_tokens (int), cost_usd (float), model (str).
    """
    cost_per_million = _COST_PER_MILLION.get(model, 1.00)  # default $1/M if unknown
    cost_usd = (token_count / 1_000_000) * cost_per_million
    return {
        "input_tokens": token_count,
        "cost_usd":     round(cost_usd, 8),
        "model":        model,
    }


# ══════════════════════════════════════════════════════
# INSTRUCTOR EXAMPLE (mock — no API key required)
# ══════════════════════════════════════════════════════

class PersonInfo(BaseModel):
    """Structured output schema for person extraction.

    Real usage with Instructor:
        import instructor, openai
        client = instructor.patch(openai.OpenAI())
        person = client.chat.completions.create(
            model="gpt-4o-mini",
            response_model=PersonInfo,
            messages=[{"role": "user", "content": text}]
        )
    """
    name: str
    age: int
    occupation: str


def extract_person_info_mock(text: str) -> PersonInfo:
    """Mock instructor extraction — in real use: instructor.patch(client).chat.completions.create(...)

    Parses name, age, and occupation from text using simple heuristics.
    This demonstrates what Instructor returns without requiring an API key.

    Args:
        text: Raw text containing person information.

    Returns:
        PersonInfo Pydantic model populated from the text.
    """
    import re

    # Extract name: look for "Name:" or first capitalised words
    name = "Unknown"
    name_match = re.search(r"(?:name[:\s]+)([A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*)", text, re.IGNORECASE)
    if name_match:
        name = name_match.group(1).strip()
    else:
        # Fall back to first two capitalised words
        cap_words = re.findall(r"\b[A-Z][a-z]+\b", text)
        if cap_words:
            name = " ".join(cap_words[:2])

    # Extract age: look for digits near "year" or "age"
    age = 0
    age_match = re.search(r"(\d{1,3})\s*(?:years? old|year-old|yr)", text, re.IGNORECASE)
    if not age_match:
        age_match = re.search(r"age[d]?\s*(?:is\s*)?(\d{1,3})", text, re.IGNORECASE)
    if not age_match:
        # any standalone number 1-99
        age_match = re.search(r"\b([1-9][0-9]?)\b", text)
    if age_match:
        age = int(age_match.group(1))

    # Extract occupation: look for "works as", "is a/an", or occupation keywords
    occupation = "Unknown"
    occ_match = re.search(
        r"(?:works? as(?: an?)?|is an?|profession[:\s]+)\s*([a-zA-Z ]{3,30})",
        text, re.IGNORECASE
    )
    if occ_match:
        occupation = occ_match.group(1).strip().rstrip(".,;")

    return PersonInfo(name=name, age=age, occupation=occupation)


# ══════════════════════════════════════════════════════
# MAIN DEMO
# ══════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("PROMPT ENGINEERING DEMO")
    print("=" * 70)

    print("\n[1] Zero-shot prompt")
    zs = zero_shot_prompt("Classify the sentiment of this movie review as positive, negative, or neutral.")
    formatted = zs.format(input="The film was absolutely brilliant — I loved every minute.")
    print(formatted)

    print("\n[2] Few-shot prompt")
    examples = [
        {"input": "The food was amazing!", "output": "positive"},
        {"input": "Terrible service, will not return.", "output": "negative"},
        {"input": "It was okay, nothing special.", "output": "neutral"},
    ]
    fs = few_shot_prompt("Classify sentiment:", examples)
    formatted = fs.format(input="Best movie of the year, highly recommend!")
    print(formatted[:300])

    print("\n[3] Chain-of-Thought prompt")
    cot = chain_of_thought_prompt("Solve this maths problem.")
    formatted = cot.format(input="If a train travels 120 km in 2 hours, what is its speed in m/s?")
    print(formatted)

    print("\n[4] Token counting")
    text = "The quick brown fox jumps over the lazy dog."
    n = count_tokens(text)
    print(f"  '{text}' -> {n} tokens")

    print("\n[5] Cost estimation")
    cost = estimate_cost(1000, model="gpt-4o-mini")
    print(f"  1000 tokens on gpt-4o-mini: ${cost['cost_usd']}")

    print("\n[6] Mock Instructor extraction")
    sample = "John Smith is 30 years old and works as a software engineer."
    person = extract_person_info_mock(sample)
    print(f"  Text: '{sample}'")
    print(f"  Extracted: {person.model_dump()}")


if __name__ == "__main__":
    main()
