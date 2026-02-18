import os
import sys
import json
from typing import List, Optional, TypedDict

from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END
import chromadb

# --- Configuration ---
# We use OpenAI for both the LLM and Embeddings for stability
llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

chroma_client = chromadb.Client()
try:
    chroma_client.delete_collection("issue_tracker")
except:
    pass

collection = chroma_client.create_collection(
    name="issue_tracker",
    metadata={"hnsw:space": "cosine"} 
)

# --- Data Models ---
class IssueStructure(BaseModel):
    needs_more_info: bool = Field(description="True if the issue is vague, False if it has technical details")
    repro_steps: List[str] = Field(description="List of steps to reproduce the issue")
    expected: str = Field(description="What the user expected to happen")
    actual: str = Field(description="What actually happened")
    # CHANGED: Changed from Dict to str to avoid OpenAI Strict Schema errors
    environment: Optional[str] = Field(default=None, description="Environment details like Browser, OS, Device")

class Classification(BaseModel):
    component: str
    severity: str
    confidence: float

class Match(BaseModel):
    issue_id: str
    score: float
    component: str
    severity: str
    status: str

class DuplicateResult(BaseModel):
    matches: List[Match]
    is_duplicate: bool
    reason: str

class RouterOutput(BaseModel):
    action: str
    target: str
    note: str

class TriageState(TypedDict):
    raw_issue: str
    structure: Optional[IssueStructure]
    classification: Optional[Classification]
    duplicate_result: Optional[DuplicateResult]
    final_decision: Optional[RouterOutput]
    retry_count: int

# --- Helper Functions ---

def seed_database():
    if collection.count() > 0: return
    try:
        with open("seed_issues.json", "r") as f:
            seed_data = json.load(f)
        ids = [issue["id"] for issue in seed_data]
        documents = [f"{issue['summary']} {issue['description']}" for issue in seed_data]
        metadatas = [{"component": issue["component"], "severity": issue["severity"], "status": "open"} for issue in seed_data]
        vectors = embeddings.embed_documents(documents)
        collection.add(ids=ids, embeddings=vectors, metadatas=metadatas, documents=documents)
        print("Database seeded with 50 issues.\n")
    except Exception as e:
        print(f"Seed error: {e}")

def increment_retry(state: TriageState):
    return {"retry_count": state["retry_count"] + 1}

def check_classifier_confidence(state: TriageState):
    if state['classification'].confidence < 0.85 and state['retry_count'] < 2:
        return "retry"
    return "confident"

# --- Node Functions ---

def extractor_agent(state: TriageState):
    print(f"--- Extractor (Attempt {state['retry_count'] + 1}) ---")
    structured_llm = llm.with_structured_output(IssueStructure)
    
    reflexion_context = ""
    if state.get('classification') and state['retry_count'] > 0:
        prev = state['classification']
        reflexion_context = f"PREVIOUS ATTEMPT FAILED: Classifier was uncertain about component '{prev.component}'. Look for deeper technical signals."

    prompt = f"""
    {reflexion_context}
    Extract structured details from: {state['raw_issue']}

    GUIDELINE: You are a lenient triage assistant. 
    Set needs_more_info=False (meaning "good to go") if the report contains ANY of the following:
    - A mention of a specific feature, page, or error message (e.g. "500 error", "payment page").
    - A description of what happened vs what was expected.
    - A rough set of steps.
    
    Only set needs_more_info=True if the report is completely empty, gibberish, or purely emotional.
    """
    res = structured_llm.invoke(prompt)
    return {"structure": res}

def classifier_agent(state: TriageState):
    print("--- Classifier ---")
    structured_llm = llm.with_structured_output(Classification)
    
    prompt = f"""
    Classify this bug report into exactly one component.
    
    CONFIDENCE SCORING RULES:
    1. HIGH (0.9 - 1.0): Report has specific error codes (500, 403), precise technical terms, or exact repro steps.
    2. MEDIUM (0.7 - 0.8): Report describes a feature but lacks technical details.
    3. LOW (0.1 - 0.6): Report uses vague words like "weird", "slow", "glitchy", "broken layout", or just says something "doesn't work".
    
    COMPONENT RULES:
    - If a UI element is broken, it's Frontend.
    - Only use Payments if the actual payment processing logic is broken.

    COMPONENTS: [Frontend, Backend, API, Mobile, Payments, Database, Auth, Infrastructure, Search, Security, Marketing]

    Issue: {state['structure'].actual}
    Repro Steps: {state['structure'].repro_steps}
    """
    res = structured_llm.invoke(prompt)
    print(f"      [DEBUG] Component: {res.component}, Confidence: {res.confidence}")
    return {"classification": res}

def duplicate_checker(state: TriageState):
    print("--- Duplicate Checker ---")
    query_text = f"{state['structure'].actual} {' '.join(state['structure'].repro_steps)}"
    query_vec = embeddings.embed_query(query_text)
    
    results = collection.query(query_embeddings=[query_vec], n_results=1)
    
    if not results['ids'][0]:
        return {"duplicate_result": DuplicateResult(matches=[], is_duplicate=False, reason="No database entries.")}

    dist = results['distances'][0][0]
    meta = results['metadatas'][0][0]
    score = 1.0 - dist 
    
    current_comp = state['classification'].component.lower()
    match_comp = meta['component'].lower()
    
    print(f"      [DEBUG] Match: {results['ids'][0][0]} | Score: {score:.4f} | Comp: {match_comp} vs {current_comp}")

    # --- FIXED LOGIC START ---
    # 1. High score: duplicate regardless of component
    strong_match = score > 0.60
    
    # 2. Moderate score + Component match: duplicate
    # Added (score > 0.45) to prevent false positives on same-component issues
    component_match = (current_comp == match_comp) and (score > 0.45)
    
    is_duplicate = strong_match or component_match
    # --- FIXED LOGIC END ---
    
    reason = f"Score {score:.2f} too low"
    if is_duplicate:
        reason = f"Confirmed duplicate of {results['ids'][0][0]} (Score {score:.2f})"

    match_obj = Match(issue_id=results['ids'][0][0], score=score, component=match_comp, severity=meta['severity'], status=meta['status'])
    return {"duplicate_result": DuplicateResult(matches=[match_obj], is_duplicate=is_duplicate, reason=reason)}

def router_agent(state: TriageState):
    print("--- Router ---")
    if state['structure'].needs_more_info:
        decision = RouterOutput(action="request_info", target="User", note="Insufficient data.")
    elif state['duplicate_result'].is_duplicate:
        decision = RouterOutput(action="mark_duplicate", target=state['duplicate_result'].matches[0].issue_id, note=state['duplicate_result'].reason)
    else:
        decision = RouterOutput(action="assign", target=f"{state['classification'].component} Team", note="New issue.")
    return {"final_decision": decision}

# --- Graph Setup ---
workflow = StateGraph(TriageState)
workflow.add_node("extractor", extractor_agent)
workflow.add_node("classifier", classifier_agent)
workflow.add_node("dup_checker", duplicate_checker)
workflow.add_node("router", router_agent)
workflow.add_node("increment_retry", increment_retry)

workflow.set_entry_point("extractor")

workflow.add_conditional_edges("extractor", lambda x: "router" if x["structure"].needs_more_info else "classifier")
workflow.add_conditional_edges("classifier", check_classifier_confidence, {"retry": "increment_retry", "confident": "dup_checker"})
workflow.add_edge("increment_retry", "extractor")
workflow.add_edge("dup_checker", "router")
workflow.add_edge("router", END)

app = workflow.compile()

# --- Runner ---
if __name__ == "__main__":
    seed_database()
    
    try:
        with open("test_issues.json", "r") as f:
            test_cases = json.load(f)
    except FileNotFoundError:
        print("Error: Please create test_issues.json")
        sys.exit(1)

    print(f"{'GROUP':<25} | {'ACTION':<15} | {'NOTE'}")
    print("-" * 110)

    for case in test_cases:
        issue_text = case.get("text") or case.get("description")
        if not issue_text: continue
            
        res = app.invoke({"raw_issue": issue_text, "retry_count": 0})
        decision = res["final_decision"]
        group = case.get("test_group", "Test")
        print(f"{group:<25} | {decision.action:<15} | {decision.note}")