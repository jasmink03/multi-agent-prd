import os
from typing import Literal, Dict, Any, List
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ────────────────────────────────────────────────────────
# 🛠️ ENHANCEMENT: READ ENV VARIABLES FROM YOUR .env FILE
# ────────────────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()  # This reads the .env file and loads GITHUB_TOKEN="xxxxxx" into os.environ

# ==========================================
# 1. CORE DEFINITIONS & CONFIGURATIONS
# ==========================================

WRITER_SYSTEM_PROMPT = """You are a premier Product Management Writer Agent. 
Transform messy notes into a deeply structured Markdown PRD containing exactly:
1. Title, 2. Executive Summary, 3. User Stories, 4. Technical Specifications, 5. Edge Cases, 6. Success Metrics."""

CRITIC_SYSTEM_PROMPT = """You are a strict QA Compliance Reviewer Agent. 
Audit the provided PRD draft against our 6-section template rubric."""

# Read token directly from the injected environment variables
github_token = os.environ.get("GITHUB_TOKEN")

if not github_token:
    raise ValueError("System Missing Credentials: GITHUB_TOKEN not found in your environment or .env file.")

# Initialize the Base LLM Engine
llm = ChatOpenAI(
    model="gpt-4o",
    api_key=github_token,  # Plugs your token from the .env directly
    base_url="https://models.inference.ai.azure.com",
    temperature=0.2
)



# ==========================================
# 2. WEEK 3: STRUCTURED OUTPUT & SCHEMAS
# ==========================================

class AgentWorkflowState(BaseModel):
    initial_input: str = ""
    current_draft: str = ""
    critic_feedback: str = ""
    critic_score: int = 0
    revision_count: int = 0
    max_revisions: int = 3
    human_approved: bool = False  # Added for Week 4 Human-in-the-Loop tracking

class CriticEvaluation(BaseModel):
    """Structured output schema forcing the LLM to output predictable JSON validation."""
    score: int = Field(description="Rubric score from 0-100 based on completeness.")
    feedback: str = Field(description="Bullet-point tracking of missing items or formatting alignment gaps.")
    is_passing: bool = Field(description="True if score is >= 80, otherwise False.")

# ==========================================
# 3. WORKFLOW AGENT NODES
# ==========================================

def writer_agent(state: AgentWorkflowState) -> Dict[str, Any]:
    print(f"\n[Writer Node] Generating/Amending Draft (Revision: {state.revision_count + 1})...")
    
    if not state.current_draft:
        user_content = f"Draft an extensive structural PRD from these base inputs:\n\n{state.initial_input}"
    else:
        user_content = f"Current Draft:\n{state.current_draft}\n\nStrict Critique to Implement:\n{state.critic_feedback}"
        
    messages = [SystemMessage(content=WRITER_SYSTEM_PROMPT), HumanMessage(content=user_content)]
    response = llm.invoke(messages)
    return {"current_draft": response.content, "revision_count": state.revision_count + 1}

def critic_agent(state: AgentWorkflowState) -> Dict[str, Any]:
    print("[Critic Node] Structuring output and auditing document compliance...")
    
    # WEEK 3 CRITICAL STEP: Binding Pydantic schemas via structured tool-calling APIs
    structured_llm = llm.with_structured_output(CriticEvaluation)
    
    messages = [SystemMessage(content=CRITIC_SYSTEM_PROMPT), HumanMessage(content=state.current_draft)]
    evaluation = structured_llm.invoke(messages)
    
    print(f"[Critic Node] Analysis Results -> Score: {evaluation.score}/100 | Target Met: {evaluation.is_passing}")
    return {"critic_feedback": evaluation.feedback, "critic_score": evaluation.score}

def human_review_node(state: AgentWorkflowState) -> Dict[str, Any]:
    """
    A placeholder node execution. In LangGraph, when we mark this node as an interrupt,
    the workflow will pause *before* running this block, waiting for external API input.
    """
    print("[Human Review Node] State unfrozen. Human feedback applied.")
    return {}

# ==========================================
# 4. WEEK 4: LANGGRAPH ORCHESTRATION WITH INTERRUPTS
# ==========================================

def routing_logic(state: AgentWorkflowState) -> Literal["writer", "end"]:
    # Loop-break condition if maximum cycles are breached
    if state.revision_count >= state.max_revisions:
        print("[Router] Maximum revision threshold reached. Forcing route to endpoint.")
        return "end"
        
    # The graph can only complete if the auto-critic passes AND the human flags it true
    if state.critic_score >= 80 and state.human_approved:
        print("[Router] Critique and Human approval conditions validated successfully. Completing.")
        return "end"
    
    print(f"[Router] Target conditions unmet (Score: {state.critic_score}/100, Approved: {state.human_approved}). Recycler active...")
    return "writer"

# Constructing Workflow Architecture
workflow = StateGraph(AgentWorkflowState)
workflow.add_node("writer", writer_agent)
workflow.add_node("critic", critic_agent)
workflow.add_node("human_review", human_review_node)

workflow.set_entry_point("writer")
workflow.add_edge("writer", "critic")

# Transition from automated critique into our Human Gate Node
workflow.add_edge("critic", "human_review")

# Transition out from Human Gate via our verification router logic
workflow.add_conditional_edges("human_review", routing_logic, {"writer": "writer", "end": END})

# WEEK 4 CRITICAL STEP: Initialize a memory checkpointer and define compile interrupts
memory_checkpointer = MemorySaver()
app = workflow.compile(
    checkpointer=memory_checkpointer,
    interrupt_before=["human_review"]  # State execution freezes completely before entering this block
)

# ==========================================
# 5. WEEK 4: FASTAPI BACKEND API WRAPPING
# ==========================================

api_service = FastAPI(
    title="Agentic Document Review Pipeline API", 
    description="Production-ready asynchronous multi-agent system built with LangGraph and FastAPI."
)

api_service.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DesignSubmissionRequest(BaseModel):
    task_id: str
    raw_notes: str

class HumanApprovalPayload(BaseModel):
    task_id: str
    approve: bool
    override_feedback: str = ""

@api_service.post("/api/v1/prd/generate", status_code=202)
async def initialize_document_pipeline(payload: DesignSubmissionRequest):
    """Endpoint to trigger the asynchronous agent workflow."""
    config = {"configurable": {"thread_id": payload.task_id}}
    initial_payload = {"initial_input": payload.raw_notes}
    
    try:
        # Run workflow up until the human gate interrupt point
        app.invoke(initial_payload, config=config)
        
        # Pull current state snapshot to send back context to the reviewer interface
        current_state = app.get_state(config)
        state_values = current_state.values
        
        return {
            "status": "Awaiting_Human_Approval",
            "task_id": payload.task_id,
            "automated_score": state_values.get("critic_score"),
            "automated_feedback": state_values.get("critic_feedback"),
            "current_draft": state_values.get("current_draft")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline Orchestration Fault: {str(e)}")

@api_service.post("/api/v1/prd/approve")
async def process_human_gate_response(payload: HumanApprovalPayload):
    """Endpoint to resume the state matching the thread_id after human feedback."""
    config = {"configurable": {"thread_id": payload.task_id}}
    
    current_snapshot = app.get_state(config)
    if not current_snapshot.values:
        raise HTTPException(status_code=404, detail="Requested generation pipeline state context not found.")
        
    # Inject human validation updates direct to state registry values
    updated_values = {
        "human_approved": payload.approve,
        "critic_feedback": payload.override_feedback if not payload.approve else current_snapshot.values.get("critic_feedback")
    }
    
    # Commit variables into the persisted graph checkpoint tracking history
    app.update_state(config, updated_values, as_node="human_review")
    
    # Resume agent pipeline processing
    final_output_state = app.invoke(None, config=config)
    
    if final_output_state.get("revision_count", 0) >= final_output_state.get("max_revisions", 3) and not final_output_state.get("human_approved"):
        return {
            "status": "Terminated_With_Faults",
            "message": "System timed out across maximum safe editing parameters without matching core criteria.",
            "data": final_output_state
        }
        
    if not payload.approve:
        return {
            "status": "Returned_To_Writer_For_Revisions",
            "automated_score": final_output_state.get("critic_score"),
            "critic_feedback": final_output_state.get("critic_feedback"),
            "current_draft": final_output_state.get("current_draft")
        }
        
    return {
        "status": "Execution_Complete_Approved",
        "final_product_requirement_document": final_output_state.get("current_draft")
    }