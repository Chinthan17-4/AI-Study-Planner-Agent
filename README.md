# AI Study Planner Agent

An **AI-powered study planning assistant** that combines a **trained ML model** with **LLM-based natural language understanding** to give students personalized exam preparation recommendations.

---

## Problem Being Solved

Students struggle to decide:
- How many hours to study for a subject
- Which subjects to prioritize when multiple exams are approaching

This agent takes a student's situation (subject, difficulty, previous score, days left, syllabus completion) and provides **data-driven study recommendations** through natural conversation.

---

## Architecture

```
Traditional ML:
  Dataset → Train Model → Prediction

This project:
  User → LLM → Tool Selection → Python Tools → ML Calculation → LLM → Final Recommendation
```

### Key Insight

The **LLM (GPT-OSS-20B)** is **not** the ML study-hours model. The LLM acts as a **reasoning layer** that:
1. Understands the user's natural language question
2. Decides which tool(s) to call
3. Extracts parameters from the conversation
4. Asks for missing information
5. Sends tool results back through the LLM for a human-friendly answer

The **ML model (RandomForestRegressor)** is a separate, pre-trained model that performs the actual numerical prediction of study hours.

### Data Flow

```
┌──────────────────────────────────────────────────────┐
│                  User (Terminal)                      │
│  "My AI exam is in 2 days, I scored 45 last time"    │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│              Groq API (GPT-OSS-20B)                  │
│  Understands question, decides to call tools,        │
│  asks for missing info (completion %, difficulty)     │
└──────────────┬───────────────────────────────────────┘
               │  tool_calls with arguments
               ▼
┌──────────────────────────────────────────────────────┐
│              Python Tool Dispatcher                  │
│  Validates arguments → Executes Python functions     │
│                                                      │
│  ┌─────────────────────┐  ┌───────────────────────┐  │
│  │ predict_study_time  │  │ prioritize_subject    │  │
│  │ (ML model .pkl)     │  │ (Rule-based formula)  │  │
│  └─────────────────────┘  └───────────────────────┘  │
└──────────────┬───────────────────────────────────────┘
               │  tool results (JSON)
               ▼
┌──────────────────────────────────────────────────────┐
│              Groq API (GPT-OSS-20B)                  │
│  Reads tool results, generates recommendation       │
└──────────────┬───────────────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────────────┐
│              Final Natural-Language Answer            │
│  "You should study ~6.78 hours. Priority: HIGH..."   │
└──────────────────────────────────────────────────────┘
```

---

## Dataset

**File:** `data/study_data_5000_10subjects.csv`

- 5,000 synthetic student-study records
- **Columns:** `subject`, `difficulty`, `previous_score`, `days_left`, `completion_percent`, `study_hours`
- **Subjects:** AI, DBMS, Maths, OS, Python, Data Structures, Computer Networks, Web Development, Software Engineering, Statistics

The dataset is used to train the ML model. It is **not** used at runtime.

---

## ML Model

**File:** `models/study_hours_model.pkl`

- **Algorithm:** RandomForestRegressor (scikit-learn)
- **Features:** `difficulty`, `previous_score`, `days_left`, `completion_percent`
- **Target:** `study_hours`
- **Training script:** `src/train_model.py` (run manually if you need to retrain)

---

## Tools

### 1. `predict_study_time`

Predicts recommended study hours using the trained ML model.

| Parameter | Type | Description |
|---|---|---|
| `difficulty` | int | Subject difficulty (1-5) |
| `previous_score` | float | Previous score (0-100) |
| `days_left` | int | Days until exam |
| `completion_percent` | float | Syllabus completed (0-100) |

**Returns:** `{"recommended_hours": 6.78}`

### 2. `prioritize_subject`

Calculates subject priority using a weighted formula (credits × 0.25 + urgency × 0.40 + weakness × 0.35).

| Parameter | Type | Description |
|---|---|---|
| `credits` | int | Academic credits (1-5) |
| `days_left` | int | Days until exam |
| `previous_marks` | float | Previous marks (0-100) |

**Returns:** `{"priority_score": 0.78, "priority": "HIGH"}`

---

## LLM

- **Provider:** Groq
- **Model:** `openai/gpt-oss-20b`
- The LLM is called via the Groq Python SDK using the chat completions API with tool schemas

---

## Agent Loop

The agent implements an **iterative tool-calling loop**:

```python
# Pseudocode
user_message → add to messages

while True:
    response = call_llm(messages)

    if response has tool_calls:
        for each tool_call:
            validate arguments
            execute tool
            add result to messages
        continue  # call LLM again with tool results
    else:
        display final answer
        break
```

This means the LLM can:
- Call one tool, get results, then call another tool
- Call multiple tools in a single turn
- Ask for missing information, then call tools on a follow-up turn

---

## Conversation Memory

The agent maintains an in-memory `messages` list that stores all conversation turns:
- System prompt
- User messages
- Assistant messages (including tool requests)
- Tool results

This allows multi-turn conversations where the user provides information across several messages.

---

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd "AI Study Agent"
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib joblib groq python-dotenv
```

### 5. Configure `.env`

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get your API key from [https://console.groq.com/](https://console.groq.com/)

---

## How to Run

```bash
python src/llm_agent.py
```

### Debug Mode

In `src/llm_agent.py`, set:

```python
DEBUG_MODE = True   # Shows tool requests and results
DEBUG_MODE = False  # Shows only the agent's final answers
```

---

## Example Interaction

```
==================================================
        AI STUDY PLANNER AGENT
==================================================
  Ask me about study planning and exam prep.
  Type 'exit' or 'quit' to leave.

You: My AI exam is in 2 days and I scored 45 last time.

Agent: I'd like to help you plan! Could you tell me:
       1. How much of the syllabus have you completed (as a percentage)?
       2. How difficult would you rate the subject from 1 to 5?

You: I've completed about 30%, and it's very hard — I'd say 5.

  --- TOOL REQUEST ---
  Tool:      predict_study_time
  Arguments: {'difficulty': 5, 'previous_score': 45, 'days_left': 2, 'completion_percent': 30}
  --- TOOL RESULT ---
  {'recommended_hours': 6.78}

Agent: Based on the ML model's prediction, you should study approximately
       **6.78 hours** for your AI exam. Given only 2 days left with 30%
       completion and a difficulty of 5, I'd recommend starting immediately
       with the most challenging topics first.

You: exit

Agent: Goodbye! Good luck with your studies!
```

---

## Project Structure

```
AI Study Agent/
│
├── data/
│   └── study_data_5000_10subjects.csv   # Training dataset (5,000 records)
│
├── models/
│   └── study_hours_model.pkl            # Trained RandomForestRegressor
│
├── src/
│   ├── predictor.py                     # ML model inference function
│   ├── priority_tool.py                 # Priority calculation logic
│   ├── tools.py                         # Tool wrapper functions
│   ├── llm_agent.py                     # Main agent (entry point)
│   ├── train_model.py                   # Model training script
│   ├── agent.py                         # Simple non-LLM agent demo
│   ├── explore_data.py                  # Dataset exploration script
│   └── llm_test.py                      # Basic LLM connectivity test
│
├── .env                                 # API key (not committed)
├── .gitignore
└── README.md
```

---

## Important Limitations

1. **Synthetic dataset** — The training data is artificially generated. The ML predictions should not be treated as scientifically validated personalized educational advice.

2. **No persistent storage** — Conversation history exists only in memory for the current session. Closing the program clears all context.

3. **Single-user** — Designed for one student at a time in a terminal session.

4. **Model accuracy** — The RandomForestRegressor is trained on synthetic patterns. Real-world study hour recommendations depend on many more factors.

5. **LLM dependency** — Requires an active internet connection and a valid Groq API key.

---

## Key Concepts for Interview Preparation

| Concept | Where It Appears |
|---|---|
| **Agentic AI** | LLM autonomously decides when to call tools vs. ask questions |
| **Tool/Function Calling** | LLM returns structured `tool_calls` that Python executes |
| **Iterative Reasoning** | While-loop lets LLM chain multiple tool calls |
| **ML Inference** | Pre-trained `.pkl` model loaded by `joblib` for prediction |
| **Prompt Engineering** | System prompt constrains LLM behavior and tool usage |
| **Conversation Memory** | Messages list accumulates context across turns |
| **Separation of Concerns** | Tools, ML model, priority logic, and LLM agent are separate modules |
| **Server-side Validation** | Arguments checked in Python before tool execution, even though the LLM schema also specifies `required` |
