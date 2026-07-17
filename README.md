# Multi-Agent PRD Generation Graph Pipeline

An automated, multi-agent AI system designed to transform unstructured product notes and engineering requirements into fully structured, compliant **Product Requirement Documents (PRDs)** using an iterative feedback loop.

This project demonstrates the complete engineering lifecycle of production-grade Generative AI workflows:
* **Week 1:** Prompt Engineering, Personas, and Few-Shot Structural Templates.
* **Week 2:** State Management and Graph Orchestration using **LangGraph**.
* **Week 3:** Structured Output Validation using **Pydantic**.
* **Week 4:** FastAPI Integration, Human-in-the-Loop Interrupts, and HTML Rendering.

---

## 🛠️ System Architecture

The pipeline uses a stateful, cyclic workflow where specialized agents collaborate to refine the document until it meets strict quality standards:

1. **Writer Agent:** Takes raw inputs or structural criticism and builds/revises a comprehensive Markdown PRD across 6 mandatory corporate sections.
2. **Critic Agent:** Inspects the draft via a strict evaluation rubric and generates a structured Pydantic compliance report containing a quality score (0-100) and actionable feedback.
3. **Human Gate Node:** Pauses execution state using LangGraph compilation interrupts. The workflow halts completely until a human reviewer either submits custom feedback or signs off on the document.
4. **Router (Circuit Breaker):** Checks the evaluation score and human approval. If the PRD scores **>= 80** and has human validation, or reaches the maximum revision threshold, the loop terminates; otherwise, it passes the feedback back to the Writer for a targeted update.

---

## 🚀 Getting Started

### 🗂️ Project Structure
To keep your development environment organized, your repository is structured as follows:

```text
.
├── main.py                # FastAPI Service, Routes, Viewers, and Graph Definition
├── README.md              # Project documentation and architectural overview
└── .env                   # Local environment variable configuration