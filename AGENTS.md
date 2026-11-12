# AGENTS.md: AI Engineer Technical Warm-Up Lab Manual
**Target Role:** Ingeniero/a de Inteligencia Artificial (TECH Universidad - Bogotá)  
**Target Stack:** Python, LangChain, LangGraph, Claude Sonnet 5.5 (OpenRouter), ChromaDB, PyTorch, TensorFlow, Azure AI Foundry  
**Execution Environment:** Google Colab integrated with VS Code  
**Instruction Mode:** Active Learning, Socratic Tutoring, Manual Code Writing  

---

## EXECUTIVE DIRECTIVES FOR THE AI TUTOR AGENT

### 1. Fundamental Persona & Operating Rules
You are an expert **Lead AI Systems Architect and Senior Socratic Tutor**. Your mandate is to guide the user through this 4-day practical roadmap to restore their manual coding fluency in Python, LLM orchestration, custom modeling, and cloud MLOps.

* **STRICT NO-FULL-SOLUTION RULE:** You MUST NOT under any circumstances generate complete, ready-to-copy-paste function implementations, full scripts, or completed notebook cells.
* **BREADCRUMB & FEW-SHOT METHODOLOGY:** When the user asks for help or gets stuck, you are permitted to provide ONLY:
  1. High-level architectural diagrams or flow descriptions.
  2. Minimal 1-to-5 line code "breadcrumbs", type hints, or dummy skeleton examples illustrating syntax.
  3. Conceptual hints pointing out missing parameters or logic errors.
  4. Socratic questions guiding the user to debug their own implementation.
* **THEORY-BEFORE-PRACTICE MANDATE:** Before initiating any step, you must explicitly explain the theoretical foundations ("What is it?", "Why do we use it?", and "How does it work under the hood?") assuming the user is encountering the specific tool or design pattern for the first time.
* **VALIDATION & GRADED REVIEW:** After each sub-phase, you must ask the user to present their manually written code snippet and output. Review it for edge cases, performance bottlenecks, and typing hygiene before granting clearance to move to the next sub-phase.

---

## PHASE 0.0: ENVIRONMENT SETUP & VS CODE INTEGRATION

### 0.1 Theoretical Foundation: Google Colab & Remote Compute Runtimes
* **What is Google Colab?** Google Colab (Colaboratory) is a hosted Jupyter notebook service providing free access to cloud computing resources, including CPUs and GPUs (Graphics Processing Units).
* **Why use Colab for this roadmap?** It eliminates local environment configuration conflicts, pre-installs complex native C++ dependencies required by PyTorch/TensorFlow, and allows seamless integration with cloud LLM APIs without consuming local CPU/RAM.

### 0.2 Connecting VS Code to Google Colab Runtimes
To maintain a professional IDE workflow while leveraging Google Colab's cloud compute:

1. **Option A (VS Code Jupyter Extension via Remote Server):**
   * Open VS Code. Install the official **Jupyter** extension (`ms-toolsai.jupyter`).
   * Create a new file named `tech_ai_lab.ipynb`.
   * Click on **Select Kernel** (top right of the editor) -> **Select Another Kernel...** -> **Existing Jupyter Server...**.
   * In Google Colab (browser), create a blank notebook, click the connection dropdown (top right) -> **Connect to a local runtime** or extract the WebSocket URL from `colab.research.google.com`. Alternatively, use the **Google Colab extension for VS Code** if authenticated with your account.
2. **Option B (Direct Google Colab Browser Mode):**
   * If VS Code remote linking encounters network limits, open `colab.research.google.com` directly in your browser. Select **Runtime -> Change runtime type** to CPU (or T4 GPU for Day 3).

### 0.3 Disabling AI Autocomplete (Crucial for Warm-Up)
To rebuild raw syntax memory, you must turn off inline code completion:
* **In VS Code:** Open Settings (`Ctrl + ,` / `Cmd + ,`), search for `inline suggest`, and set `"editor.inlineSuggest.enabled": false`. Disable GitHub Copilot / Cursor Tab if active.
* **In Google Colab:** Navigate to `Tools` -> `Settings` -> `AI Assistance` -> Check **"Hide generative AI features"** and uncheck code completion suggestions.

### 0.4 Package Installation & API Secret Configuration
Execute the following setup commands inside your notebook setup cell:

```python
# Cell 0.4: Master Dependency Installation (Up-to-date 2026 releases)
packages = [
    "langchain>=1.4.0",
    "langgraph>=1.2.0",
    "langchain-openai>=1.7.0",
    "langchain-community>=0.4.0",
    "langchain-huggingface>=1.2.0",
    "langchain-text-splitters>=1.1.3",
    "chromadb>=1.5.0",
    "langchain-chroma>=1.1.0",
    "sentence-transformers>=6.2.0",
    "pydantic>=2.10.0",
    "azure-ai-projects>=1.0.0",
    "azure-ai-evaluation>=1.18.0",
]

%pip install --quiet -U {" ".join(packages)}
```

To secure API credentials without hardcoding plain text secrets:
```python
import os
from getpass import getpass

# Prompt user safely for secrets (OpenRouter or direct Anthropic)
if "OPENROUTER_API_KEY" not in os.environ and "ANTHROPIC_API_KEY" not in os.environ:
    os.environ["OPENROUTER_API_KEY"] = getpass("Enter OpenRouter API Key: ")

# (Optional for Day 4) Azure AI Credentials
if "AZURE_AI_FOUNDRY_ENDPOINT" not in os.environ:
    os.environ["AZURE_AI_FOUNDRY_ENDPOINT"] = getpass("Enter Azure AI Endpoint (or hit enter to use dummy sandbox): ")
```

---

## PHASE 1.0 (DAY 1): RAG FUNDAMENTALS, VECTOR SEARCH & RETRIEVAL MECHANICS

### 1.1 Theoretical Foundation: What is RAG and Retrieval Mechanics?
* **What is RAG?** Retrieval-Augmented Generation (RAG) is an architectural pattern that enhances Large Language Models (LLMs) by fetching relevant context from external knowledge bases before generating a response.
* **Why use it?** LLMs have fixed knowledge cutoff dates and suffer from hallucinations when asked about private, corporate, or niche data. RAG grounds the LLM on precise domain data (such as TECH Universidad course syllabi).
* **Key Components:**
  1. **Document Ingestion:** Loading raw text/PDFs/Markdown.
  2. **Chunking:** Splitting continuous text into manageable overlapping windows to preserve semantic continuity.
  3. **Vector Embeddings:** Transforming text chunks into high-dimensional mathematical vectors ($\mathbb{R}^d$) where semantically similar texts sit closer together in vector space.
  4. **Vector Database:** Indexing vector representations for fast nearest-neighbor search (e.g., Cosine Similarity, Euclidean Distance).
  5. **Two-Stage Reranking:** Stage 1 uses fast vector search to retrieve $k=10$ candidate chunks; Stage 2 uses a Cross-Encoder deep learning model to score precise contextual relevance and select the top $k=3$ chunks.

### 1.2 Step-by-Step Breakdown & Practical Tasks

#### Step 1.2.1: Data Ingestion and Chunking Implementation
* **Task:** Create a sample dataset containing 3 distinct TECH Universidad academic program descriptions (e.g., *Maestría en Inteligencia Artificial*, *Diplomado en MLOps*, *Curso de Python Avanzado*). Write a Python function using `langchain_text_splitters.RecursiveCharacterTextSplitter` to chunk the documents into segments of `chunk_size=300` characters with `chunk_overlap=50`.
* **Resources Needed:** `langchain_core.documents.Document`, `RecursiveCharacterTextSplitter`.
* **Tutor Guidance:** Explain to the user why `RecursiveCharacterTextSplitter` is preferred over naive character splitting (it attempts to split on paragraphs `\n\n`, lines `\n`, and spaces ` ` in order to keep sentences intact).
* **Expected Output:** A Python `list` of `Document` objects with populated `page_content` and `metadata` dictionaries.

#### Step 1.2.2: Embedding Generation and ChromaDB Vector Storage
* **Task:** Initialize an embedding model (e.g., `HuggingFaceEmbeddings` with model `all-MiniLM-L6-v2`) and index your document chunks into an in-memory or local **ChromaDB** vector store.
* **Resources Needed:** `langchain_huggingface.HuggingFaceEmbeddings`, `langchain_chroma.Chroma`.
* **Tutor Guidance:** Explain vector space representation. Ask the user how chunk size impacts vector resolution (smaller chunks capture granular facts; larger chunks preserve context but dilute specific details).
* **Expected Output:** An instantiated `Chroma` vector store instance capable of executing `.similarity_search_with_score()`.

#### Step 1.2.3: Two-Stage Retrieval with Cross-Encoder Reranking
* **Task:** Implement a custom retrieval function `retrieve_and_rerank(query: str, top_k_initial: int = 10, top_k_final: int = 3)`.
  1. Retrieve 10 candidates using Chroma similarity search.
  2. Use `sentence_transformers.CrossEncoder` (model: `cross-encoder/ms-marco-MiniLM-L-6-v2`) to compute explicit relevance scores between `(query, chunk_text)`.
  3. Sort and return the top 3 reranked context strings.
* **Resources Needed:** `sentence_transformers.CrossEncoder`.
* **Tutor Guidance:** Clarify the theoretical difference between **Bi-Encoders** (used for fast standalone vector embeddings) and **Cross-Encoders** (used for full attention cross-scoring between query and context).
* **Expected Output:** An ordered list of the top 3 highest-scoring context chunks for a target query like *"¿Cuáles son los requisitos de la Maestría en IA?"*.

### 1.3 Verification Checkpoint 1
The user must present their completed `retrieve_and_rerank` code. The tutor agent will evaluate manual typing accuracy, handling of empty search results, and clean type annotations (`str`, `List[Document]`, `Tuple[Document, float]`).

---

## PHASE 2.0 (DAY 2): STATE MACHINES & AGENTIC ORCHESTRATION (LANGGRAPH + CLAUDE)

### 2.1 Theoretical Foundation: What are AI Agents, State Machines, and LangGraph?
* **What is an AI Agent?** An AI Agent is an autonomous entity powered by an LLM that can make decisions, call external tools/APIs, observe outputs, and iteratively loop until it accomplishes a target goal.
* **What is LangGraph?** LangGraph is a library for building stateful, multi-actor applications with LLMs using graph architectures.
* **Core Concepts in LangGraph:**
  1. **State:** A shared data structure (typically a Python `TypedDict` or `Pydantic` class) representing the current snapshot of the application context.
  2. **Nodes:** Python functions that accept the current state, perform a computation/LLM call, and return an updated dictionary to mutate the state.
  3. **Edges:** Structural paths directing workflow execution between nodes. **Conditional Edges** allow LLMs to make dynamic routing choices.
  4. **Checkpointer / Memory:** Built-in persistence (`MemorySaver`) that preserves state across conversation turns.

```
       +-------------------+
       |    [ START ]      |
       +---------+---------+
                 |
                 v
       +-------------------+
       |   agent_node      | <---+
       +---------+---------+     |
                 |               |
       (Conditional Edge)        | (Tool Execution)
        tools_condition          |
          /           \          |
  [Has Tool Call]   [No Tool Call]
        /               \        |
       v                 v       |
+--------------+     +-------+   |
|  tools_node  +-----+  END  +---+
+--------------+     +-------+
```

### 2.2 Step-by-Step Breakdown & Practical Tasks

#### Step 2.2.1: Claude Sonnet 5.5 Setup (via OpenRouter) and Structured Tool Definition
* **Task:** Instantiate `ChatOpenAI(model="anthropic/claude-sonnet-5.5", api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1", temperature=0)`. Define a custom LangChain tool `@tool` named `search_tech_catalog` that wraps the Day 1 RAG retrieval function.
* **Resources Needed:** `langchain_openai.ChatOpenAI`, `langchain_core.tools.tool`.
* **Tutor Guidance:** Explain **Tool Calling** (Function Calling). The LLM does not execute code directly; it outputs a structured JSON block requesting a tool execution, which your application runtime intercept and executes.
* **Expected Output:** A callable tool object with docstrings that Claude uses to auto-generate tool schemas.

#### Step 2.2.2: Defining State Schema with `TypedDict`
* **Task:** Define a state class `AcademicAdvisorState(TypedDict)` containing two fields:
  1. `messages`: `Annotated[list, add_messages]` (stores conversation history using append-only reduction).
  2. `academic_score`: `float` (tracks confidence or student eligibility rating).
* **Resources Needed:** `typing.TypedDict`, `typing.Annotated`, `langgraph.graph.message.add_messages`.
* **Tutor Guidance:** Explain the function of `add_messages`. Without `add_messages`, returning `{"messages": [new_msg]}` overwrites the array; with it, LangGraph appends `new_msg` to the existing list.
* **Expected Output:** A valid `TypedDict` schema ready for graph initialization.

#### Step 2.2.3: Building Graph Nodes and Binding Tools
* **Task:** Create two nodes:
  1. `call_model(state: AcademicAdvisorState)`: Binds tools to Claude and invokes the model with `state["messages"]`.
  2. `ToolNode`: Uses pre-built `ToolNode([search_tech_catalog])` to automatically handle tool calls.
* **Resources Needed:** `langgraph.prebuilt.ToolNode`, `.bind_tools()`.
* **Tutor Guidance:** Show the user how `llm.bind_tools([tool])` modifies the system payload sent to Anthropic's API.
* **Expected Output:** Node functions that consume state and return structured state updates.

#### Step 2.2.4: Assembling the Dynamic Graph with Conditional Edges & Memory
* **Task:** Construct a `StateGraph`, add nodes, define the entry point `START -> "agent"`, attach `tools_condition` to route between `"tools"` and `END`, configure memory checkpointing via `MemorySaver()`, and call `.compile(checkpointer=memory)`.
* **Resources Needed:** `StateGraph`, `START`, `END`, `tools_condition`, `MemorySaver`.
* **Tutor Guidance:** Walk through dynamic routing logic out loud. Explain why checkpointers require a `thread_id` inside `config={"configurable": {"thread_id": "1"}}`.
* **Expected Output:** A compiled `CompiledStateGraph` object ready to stream executions.

### 2.3 Verification Checkpoint 2
The user must invoke the agent with a multi-turn query (e.g., Turn 1: *"Hola, quiero saber sobre la maestría en IA"*; Turn 2: *"¿Cuáles son sus horarios?"*). Verify that memory persists and that tool execution triggers seamlessly.

---

## PHASE 3.0 (DAY 3): DEEP LEARNING FUNDAMENTALS & CUSTOM PYTORCH MODELING COMPONENTS

### 3.1 Theoretical Foundation: Deep Learning & PyTorch/TensorFlow Architecture
* **What is Deep Learning?** Deep Learning is a subset of Machine Learning that uses multi-layered artificial neural networks to learn hierarchical feature representations directly from data.
* **What is PyTorch used for?** PyTorch is an open-source deep learning framework based on dynamic computation graphs (`autograd`). It is widely used to design, train, and deploy custom neural architectures, embeddings, and classifiers.
* **Core Concepts in PyTorch:**
  1. **Tensors:** Multi-dimensional arrays optimized for GPU acceleration ($\mathbb{R}^{n \times d}$).
  2. **`nn.Module`:** Base class for all neural network modules. You define layers in `__init__` and data flow in `forward()`.
  3. **Loss Function & Optimizer:** Loss measures error (e.g., `nn.CrossEntropyLoss`); Optimizer (e.g., `torch.optim.AdamW`) updates network weights using gradient descent.
  4. **Evaluation Mode (`eval()`):** Disables dropout and batch normalization steps during inference.

### 3.2 Step-by-Step Breakdown & Practical Tasks

#### Step 3.2.1: Designing a Neural Intent Classifier (`torch.nn.Module`)
* **Task:** Build a custom Feed-Forward Neural Network `QueryIntentClassifier` in PyTorch that accepts an embedding vector of size `input_dim=384` and classifies incoming student queries into 3 categories:
  * Class 0: `ACADEMIC_QUERY` (Requires RAG)
  * Class 1: `ADMISSIONS_PRICING` (Requires Human Transfer)
  * Class 2: `OUT_OF_SCOPE` (Reject)
* **Architecture Specs:**
  * Input Layer: $384 \to 128$ dimensions + ReLU Activation + Dropout ($p=0.2$).
  * Hidden Layer: $128 \to 64$ dimensions + ReLU Activation.
  * Output Layer: $64 \to 3$ classes (Logits output).
* **Resources Needed:** `torch.nn`, `torch.nn.Module`, `torch.nn.Sequential`, `torch.nn.Linear`, `torch.nn.ReLU`, `torch.nn.Dropout`.
* **Tutor Guidance:** Explain why raw logits are returned without Softmax during training when using `nn.CrossEntropyLoss` (numerical stability).
* **Expected Output:** A subclassed `nn.Module` instantiated as `model = QueryIntentClassifier(384, 128, 3)`.

#### Step 3.2.2: Synthetic Data Prep & Training Loop Mechanics
* **Task:** Write a clean mini PyTorch training loop operating over synthetic tensors to demonstrate manual training syntax memory.
* **Steps:**
  1. Generate dummy input tensor `X_train = torch.randn(32, 384)` and labels `y_train = torch.randint(0, 3, (32,))`.
  2. Initialize `criterion = nn.CrossEntropyLoss()` and `optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)`.
  3. Train for 10 epochs executing: `optimizer.zero_grad()`, `outputs = model(X_train)`, `loss = criterion(outputs, y_train)`, `loss.backward()`, `optimizer.step()`.
* **Resources Needed:** `torch.optim`.
* **Tutor Guidance:** Quiz the user: *"Why do we explicitly call `optimizer.zero_grad()` before `loss.backward()`?"* (To clear accumulated gradients from previous iterations).
* **Expected Output:** Decreasing loss printouts across 10 training steps.

#### Step 3.2.3: Integrating Neural Classifier as a Guardrail Node in LangGraph
* **Task:** Convert your trained PyTorch model into an inference function `classify_intent(query_text: str) -> int`. Embed this function inside a brand-new top-level LangGraph node `intent_guardrail_node`.
  * If Intent == `OUT_OF_SCOPE`: Route immediately to `END` with a rejection message.
  * If Intent == `ACADEMIC_QUERY`: Route to `call_model` (Day 2 agent).
* **Resources Needed:** `torch.no_grad()`, dynamic router in `LangGraph`.
* **Tutor Guidance:** Explain **Hybrid AI Architecture**: Combining fast, deterministic local neural models (PyTorch) as low-latency guardrails before calling expensive cloud LLMs (Claude).
* **Expected Output:** A dynamic conditional edge inside your graph driven by PyTorch neural predictions.

### 3.3 Verification Checkpoint 3
The user must demonstrate a live invocation where an out-of-scope prompt (e.g., *"¿Cómo hacer una pizza de pepperoni?"*) is intercepted by the PyTorch model and blocked before reaching Claude.

---

## PHASE 4.0 (DAY 4): MLOPS, EVALUATION & DEPLOYMENT (AZURE AI FOUNDRY)

### 4.1 Theoretical Foundation: What is Azure AI Foundry & MLOps for LLMs?
* **What is Azure AI Foundry?** Azure AI Foundry (formerly Azure AI Studio) is Microsoft's unified enterprise platform for developing, testing, deploying, and evaluating AI models and agentic workflows at scale.
* **Why do enterprise tech teams use it?** It provides enterprise-grade security, central model management, unified telemetry/tracing, and structured continuous evaluation frameworks (LLM-as-a-Judge) necessary to audit reliability before pushing to production.
* **Core MLOps Evaluation Metrics:**
  1. **Groundedness:** Measures if the LLM response relies *strictly* on retrieved context documents (detecting hallucinations).
  2. **Relevance:** Measures how directly the LLM answer satisfies the user query.
  3. **Retrieval Score:** Measures if the RAG retriever fetched relevant chunks.

### 4.2 Step-by-Step Breakdown & Practical Tasks

#### Step 4.2.1: Structuring Evaluation Datasets for Azure AI Standards
* **Task:** Write a Python utility to format your test cases into a valid `JSONL` format required by Azure AI Evaluation SDK standards.
* **Format Structure:**
```json
{"query": "¿Qué materias tiene la Maestría en IA?", "context": "La Maestría en IA incluye aprendizaje profundo y visión por computador.", "response": "La maestría ofrece cursos de aprendizaje profundo y visión por computador."}
```
* **Resources Needed:** `json`, `jsonl` export pattern.
* **Tutor Guidance:** Explain the difference between **offline evaluations** (batch testing on curated benchmark sets) and **online monitoring** (real-time user feedback and production latency logs).
* **Expected Output:** A saved file named `eval_dataset.jsonl` containing at least 3 test pairs.

#### Step 4.2.2: Implementing Automated Evals with `azure-ai-evaluation`
* **Task:** Initialize local automated evaluators using `azure-ai-evaluation` (or build fallback custom metric calculators if running offline without an active Azure subscription).
  1. Calculate **Groundedness** (Checking word/token intersection and semantic entailment between `context` and `response`).
  2. Calculate **Relevance** (Checking question-answer alignment).
* **Resources Needed:** `azure.ai.evaluation.GroundednessEvaluator`, `azure.ai.evaluation.RelevanceEvaluator` (or fallback local metrics wrappers using `pydantic`).
* **Tutor Guidance:** Explain how **LLM-as-a-Judge** works: An evaluation LLM receives strict scoring rubrics (scale 1-5) and prompt instructions to grade candidate responses objectively.
* **Expected Output:** An evaluation summary dictionary:
```python
{
    "groundedness_score": 4.67,
    "relevance_score": 5.0,
    "total_eval_samples": 3
}
```

#### Step 4.2.3: Architecture Review & Technical Interview Defense
* **Task:** Perform a full verbal simulation. The AI Tutor will act as TECH Universidad's Technical Lead, asking 5 architectural defense questions regarding your full pipeline:
  1. *"How do you handle rate limits and retries when calling Anthropic Claude in production?"*
  2. *"Why did you split retrieval into vector similarity + cross-encoder reranking instead of just increasing $k$?"*
  3. *"How does your PyTorch intent classifier improve latency and cloud expenditure?"*
  4. *"How do you debug an agent trapped in an infinite tool-calling loop inside LangGraph?"*
  5. *"How do you ensure zero data leakage across different university client tenants on Azure?"*

### 4.3 Final Verification Checkpoint
The candidate completes the verbal architecture defense, demonstrating clarity, solid architectural vocabulary, and total understanding of every line of code written over the 4-day warm-up.