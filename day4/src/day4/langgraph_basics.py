"""
langgraph_basics.py — LangGraph Fundamentals
=============================================
Covers: StateGraph, TypedDict state, nodes, edges, conditional routing,
        START/END, graph compilation, .invoke() and .stream().

Core mental model:
  - STATE: a TypedDict that flows through the graph and is updated by each node
  - NODE: a Python function that receives the state and returns a partial update
  - EDGE: a connection between nodes (normal) or a function that decides the next node (conditional)
  - GRAPH: assembled with StateGraph, compiled with .compile(), run with .invoke()

Industry use:
  Every LangGraph agent at scale (LangChain, Replit, Uber, LinkedIn) follows this pattern.
  The state machine approach makes complex AI workflows predictable and debuggable.

Run: python -m day4.langgraph_basics
"""

from __future__ import annotations

from typing import TypedDict, Optional, Callable
from langgraph.graph import StateGraph, START, END


# ══════════════════════════════════════════════════════
# EXAMPLE 1: BMI Calculator Graph (pure Python, no LLM)
# ══════════════════════════════════════════════════════

class BMIState(TypedDict):
    """State for the BMI calculator graph.
    Each node receives this state dict and returns a partial update.
    """
    weight_kg: float
    height_m: float
    bmi: Optional[float]
    category: Optional[str]
    recommendation: Optional[str]


def calculate_bmi(state: BMIState) -> dict:
    """Node 1: compute BMI from weight and height."""
    bmi = state["weight_kg"] / (state["height_m"] ** 2)
    return {"bmi": round(bmi, 2)}


def classify_bmi(state: BMIState) -> dict:
    """Node 2: classify BMI into a category."""
    bmi = state["bmi"]
    if bmi < 18.5:
        category = "underweight"
    elif bmi < 25.0:
        category = "normal"
    elif bmi < 30.0:
        category = "overweight"
    else:
        category = "obese"
    return {"category": category}


def route_by_category(state: BMIState) -> str:
    """Conditional edge: route to the appropriate recommendation node."""
    return state["category"]   # returns node name


def recommend_underweight(state: BMIState) -> dict:
    return {"recommendation": "Increase caloric intake with nutrient-dense foods. Consult a nutritionist."}


def recommend_normal(state: BMIState) -> dict:
    return {"recommendation": "Great BMI! Maintain your current diet and exercise routine."}


def recommend_overweight(state: BMIState) -> dict:
    return {"recommendation": "Reduce refined carbs and increase cardio exercise 3-4x per week."}


def recommend_obese(state: BMIState) -> dict:
    return {"recommendation": "Consult a doctor for a personalised weight management programme."}


def build_bmi_graph():
    """Build and compile the BMI calculator StateGraph.

    Graph structure:
        START -> calculate_bmi -> classify_bmi -> [conditional] -> recommendation node -> END

    Returns:
        Compiled LangGraph graph ready for .invoke()
    """
    g = StateGraph(BMIState)

    g.add_node("calculate_bmi",       calculate_bmi)
    g.add_node("classify_bmi",        classify_bmi)
    g.add_node("underweight",         recommend_underweight)
    g.add_node("normal",              recommend_normal)
    g.add_node("overweight",          recommend_overweight)
    g.add_node("obese",               recommend_obese)

    g.add_edge(START,            "calculate_bmi")
    g.add_edge("calculate_bmi",  "classify_bmi")
    g.add_conditional_edges(
        "classify_bmi",
        route_by_category,
        {
            "underweight": "underweight",
            "normal":      "normal",
            "overweight":  "overweight",
            "obese":       "obese",
        }
    )
    for node in ["underweight", "normal", "overweight", "obese"]:
        g.add_edge(node, END)

    return g.compile()


# ══════════════════════════════════════════════════════
# EXAMPLE 2: Prompt Chaining Graph
# ══════════════════════════════════════════════════════

class ChainState(TypedDict):
    """State for a sequential prompt-chaining graph."""
    topic: str
    detailed_report: Optional[str]
    summary: Optional[str]
    tweet: Optional[str]


def build_prompt_chain_graph(llm_fn: Callable[[str], str]):
    """Build a 3-step prompt chaining graph: report -> summary -> tweet.

    Args:
        llm_fn: Callable(prompt: str) -> str. Pass a mock or real LLM function.

    Returns:
        Compiled LangGraph graph.
    """
    def generate_report(state: ChainState) -> dict:
        prompt = f"Write a detailed 3-paragraph report about: {state['topic']}"
        return {"detailed_report": llm_fn(prompt)}

    def summarise(state: ChainState) -> dict:
        prompt = f"Summarise this in 3 bullet points:\n{state['detailed_report']}"
        return {"summary": llm_fn(prompt)}

    def make_tweet(state: ChainState) -> dict:
        prompt = f"Write a tweet (max 280 chars) based on:\n{state['summary']}"
        result = llm_fn(prompt)
        return {"tweet": result[:280]}

    g = StateGraph(ChainState)
    g.add_node("generate_report", generate_report)
    g.add_node("summarise",       summarise)
    g.add_node("make_tweet",      make_tweet)

    g.add_edge(START,             "generate_report")
    g.add_edge("generate_report", "summarise")
    g.add_edge("summarise",       "make_tweet")
    g.add_edge("make_tweet",      END)

    return g.compile()


# ══════════════════════════════════════════════════════
# EXAMPLE 3: Review + Reply Graph (conditional branching)
# ══════════════════════════════════════════════════════

class ReviewState(TypedDict):
    """State for a review sentiment + reply graph."""
    review_text: str
    sentiment: Optional[str]     # "positive", "negative", "neutral"
    reply: Optional[str]


def analyse_sentiment(state: ReviewState) -> dict:
    """Simple rule-based sentiment (no LLM needed for demo)."""
    text = state["review_text"].lower()
    positive_words = {"great", "excellent", "love", "amazing", "good", "best", "fantastic", "wonderful"}
    negative_words = {"bad", "terrible", "awful", "worst", "hate", "poor", "horrible", "disappointing"}

    text_words = set(text.split())
    pos_score = len(text_words & positive_words)
    neg_score = len(text_words & negative_words)

    if pos_score > neg_score:
        sentiment = "positive"
    elif neg_score > pos_score:
        sentiment = "negative"
    else:
        sentiment = "neutral"
    return {"sentiment": sentiment}


def route_by_sentiment(state: ReviewState) -> str:
    return state["sentiment"]


def reply_positive(state: ReviewState) -> dict:
    return {"reply": "Thank you for your wonderful feedback! We're delighted you had a great experience."}


def reply_negative(state: ReviewState) -> dict:
    return {"reply": "We're sorry to hear this. Please contact support@company.com and we'll make it right."}


def reply_neutral(state: ReviewState) -> dict:
    return {"reply": "Thank you for your feedback! We're always working to improve."}


def build_review_graph():
    """Build a review sentiment + auto-reply graph.

    Returns:
        Compiled LangGraph graph.
    """
    g = StateGraph(ReviewState)
    g.add_node("analyse_sentiment", analyse_sentiment)
    g.add_node("positive",          reply_positive)
    g.add_node("negative",          reply_negative)
    g.add_node("neutral",           reply_neutral)

    g.add_edge(START, "analyse_sentiment")
    g.add_conditional_edges(
        "analyse_sentiment",
        route_by_sentiment,
        {"positive": "positive", "negative": "negative", "neutral": "neutral"}
    )
    for node in ["positive", "negative", "neutral"]:
        g.add_edge(node, END)

    return g.compile()


# ══════════════════════════════════════════════════════
# GRAPH INSPECTION UTILITIES
# ══════════════════════════════════════════════════════

def inspect_graph(graph) -> dict:
    """Return a dict describing the compiled graph's nodes and edges.

    Useful for tests and debugging — verify the graph structure without running it.

    Args:
        graph: A compiled LangGraph graph.

    Returns:
        Dict with keys: nodes (list), num_nodes (int).
    """
    nodes = list(graph.nodes.keys()) if hasattr(graph, "nodes") else []
    return {
        "nodes":     nodes,
        "num_nodes": len(nodes),
    }


def run_graph(graph, initial_state: dict) -> dict:
    """Run a compiled graph and return the final state.

    Args:
        graph:         Compiled LangGraph graph.
        initial_state: Starting state dict.

    Returns:
        Final state dict after all nodes have run.
    """
    return graph.invoke(initial_state)


# ══════════════════════════════════════════════════════
# MAIN DEMO
# ══════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("LANGGRAPH BASICS DEMO")
    print("=" * 70)

    print("\n[1] BMI Calculator Graph")
    bmi_graph = build_bmi_graph()
    test_cases = [
        {"weight_kg": 55, "height_m": 1.70, "bmi": None, "category": None, "recommendation": None},
        {"weight_kg": 70, "height_m": 1.75, "bmi": None, "category": None, "recommendation": None},
        {"weight_kg": 90, "height_m": 1.70, "bmi": None, "category": None, "recommendation": None},
    ]
    for case in test_cases:
        result = bmi_graph.invoke(case)
        print(f"  {result['weight_kg']}kg / {result['height_m']}m -> "
              f"BMI={result['bmi']} ({result['category']})")
        print(f"    Rec: {result['recommendation'][:70]}")

    print("\n[2] Prompt Chain Graph (mock LLM)")
    def mock_llm(prompt: str) -> str:
        if "report" in prompt.lower():
            return "India is a diverse nation with 1.4 billion people. The economy is growing at 7%. Technology and services dominate exports."
        elif "bullet" in prompt.lower() or "summarise" in prompt.lower():
            return "- India has 1.4B people\n- Economy growing 7%\n- Tech sector is leading"
        else:
            return "India: 1.4B people, 7% GDP growth, tech-led economy! #India #Economy"

    chain = build_prompt_chain_graph(mock_llm)
    result = chain.invoke({"topic": "India's economy", "detailed_report": None, "summary": None, "tweet": None})
    print(f"  Topic: {result['topic']}")
    print(f"  Tweet: {result['tweet']}")

    print("\n[3] Review Reply Graph")
    review_graph = build_review_graph()
    reviews = [
        "This product is amazing and great quality!",
        "Terrible experience, worst service ever.",
        "It was okay, nothing special.",
    ]
    for review in reviews:
        result = review_graph.invoke({"review_text": review, "sentiment": None, "reply": None})
        print(f"  [{result['sentiment'].upper()}] \"{review[:45]}\"")
        print(f"    Reply: {result['reply'][:80]}")


if __name__ == "__main__":
    main()
