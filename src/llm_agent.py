"""
AI Study Planner Agent
======================
Main entry point. Connects user input → Groq LLM → Python tools → final answer.

Architecture:
  User question
    → Groq API (GPT-OSS-20B) with tool schemas
    → LLM decides which tool(s) to call (or asks for missing info)
    → Python executes the tool(s) locally
    → Tool results sent back to LLM
    → LLM produces a natural-language recommendation

The LLM does NOT perform calculations itself — it delegates to:
  1. predict_study_time  → ML model (RandomForestRegressor)
  2. prioritize_subject  → rule-based priority formula
"""

import os
import sys
import json
from pathlib import Path

# Force UTF-8 output on Windows to handle Unicode chars in LLM responses.
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
from groq import Groq

# ---------------------------------------------------------------------------
# Resolve paths so the script works regardless of the current working dir.
# PROJECT_ROOT is the parent of src/, so .env and models/ are always found.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add src/ to Python path so that sibling module imports work.
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tools import predict_study_time, prioritize_subject  # noqa: E402


# ===========================================================================
#  CONFIGURATION
# ===========================================================================

# Set to True to see internal tool-call details in the terminal.
# Set to False to show only the agent's final answers.
DEBUG_MODE = True

# LLM model hosted on Groq.
MODEL_NAME = "openai/gpt-oss-20b"


# ===========================================================================
#  1. LOAD API KEY
# ===========================================================================

load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("\n[ERROR] GROQ_API_KEY is not set.")
    print("Create a .env file in the project root with:")
    print("  GROQ_API_KEY=your_key_here")
    sys.exit(1)

client = Groq(api_key=api_key)


# ===========================================================================
#  2. TOOL SCHEMAS  (sent to the LLM so it knows what tools exist)
# ===========================================================================

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "predict_study_time",
            "description": (
                "Predict the recommended number of study hours using "
                "the trained ML model. Requires all four parameters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "difficulty": {
                        "type": "integer",
                        "description": "Subject difficulty level from 1 (easy) to 5 (hard)."
                    },
                    "previous_score": {
                        "type": "number",
                        "description": "Student's previous score in the subject (0-100)."
                    },
                    "days_left": {
                        "type": "integer",
                        "description": "Number of days remaining before the exam."
                    },
                    "completion_percent": {
                        "type": "number",
                        "description": "Percentage of syllabus already completed (0-100)."
                    }
                },
                "required": [
                    "difficulty",
                    "previous_score",
                    "days_left",
                    "completion_percent"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "prioritize_subject",
            "description": (
                "Calculate the study priority of a subject based on "
                "credits, days remaining, and previous marks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "credits": {
                        "type": "integer",
                        "description": "Number of academic credits for the subject (1-5)."
                    },
                    "days_left": {
                        "type": "integer",
                        "description": "Number of days remaining before the exam."
                    },
                    "previous_marks": {
                        "type": "number",
                        "description": "Student's previous marks in the subject (0-100)."
                    }
                },
                "required": [
                    "credits",
                    "days_left",
                    "previous_marks"
                ]
            }
        }
    }
]


# ===========================================================================
#  3. TOOL DISPATCHER  (maps tool names → Python functions + validation)
# ===========================================================================

# Required parameters for each tool — used for server-side validation so
# we never call a Python function with missing/null arguments, even if the
# LLM hallucinates incomplete arguments.

TOOL_REQUIRED_PARAMS = {
    "predict_study_time": ["difficulty", "previous_score", "days_left", "completion_percent"],
    "prioritize_subject": ["credits", "days_left", "previous_marks"],
}

# Map of tool name → callable Python function.
TOOL_FUNCTIONS = {
    "predict_study_time": predict_study_time,
    "prioritize_subject": prioritize_subject,
}


def execute_tool(tool_name, arguments):
    """
    Validate arguments and execute a tool.

    Returns a dict — either the tool's result or a structured error.
    This function never raises; all errors are caught and returned as dicts
    so the LLM can read the error and recover gracefully.
    """

    # --- Unknown tool ---
    if tool_name not in TOOL_FUNCTIONS:
        return {"error": f"Unknown tool '{tool_name}'. Available tools: {list(TOOL_FUNCTIONS.keys())}"}

    # --- Validate required arguments ---
    required = TOOL_REQUIRED_PARAMS[tool_name]
    missing = [p for p in required if arguments.get(p) is None]

    if missing:
        return {
            "error": "Missing required arguments. Do NOT guess — ask the user.",
            "missing_fields": missing
        }

    # --- Execute the tool ---
    try:
        func = TOOL_FUNCTIONS[tool_name]
        result = func(**{p: arguments[p] for p in required})
        return result
    except Exception as exc:
        return {"error": f"Tool execution failed: {exc}"}


# ===========================================================================
#  4. SYSTEM PROMPT
# ===========================================================================

SYSTEM_PROMPT = """\
You are an AI Study Planner Agent.

You help students plan their exam preparation by using two calculation tools:

1. **predict_study_time** — Uses a trained ML model to predict how many
   hours a student should study. Requires: difficulty, previous_score,
   days_left, completion_percent.

2. **prioritize_subject** — Calculates a priority score and level (HIGH /
   MEDIUM / LOW) for a subject. Requires: credits, days_left, previous_marks.

RULES YOU MUST FOLLOW:
- You MUST use the tools when calculations are needed. Never invent numbers.
- NEVER call a tool until you have ALL of its required arguments.
- If any required information is missing, ask the user for it instead.
- Remember information the user provided in earlier messages.
- After receiving tool results, present a clear, helpful recommendation.
- Be concise and friendly.
- When the user mentions a subject, difficulty, score, days left, or
  completion percentage, map them to the correct tool parameters.
"""


# ===========================================================================
#  5. HELPER — call the Groq API (with error handling)
# ===========================================================================

def call_llm(messages):
    """Send messages to the Groq API and return the assistant message."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
        )
        return response.choices[0].message
    except Exception as exc:
        # Return None and let the caller handle it.
        print(f"\n[ERROR] Groq API call failed: {exc}")
        return None


# ===========================================================================
#  6. AGENT LOOP
# ===========================================================================

def run_agent():
    """
    Interactive agent loop.

    Maintains conversation history in `messages` (list of dicts).
    Implements an iterative tool-calling loop:
      while LLM requests tools → execute → feed results back → call LLM again
      when LLM stops requesting tools → display final answer
    """

    # Conversation memory — persists across turns within this session.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("\n" + "=" * 50)
    print("        AI STUDY PLANNER AGENT")
    print("=" * 50)
    print("  Ask me about study planning and exam prep.")
    print("  Type 'exit' or 'quit' to leave.\n")

    while True:

        # --- Get user input ---
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nAgent: Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("\nAgent: Goodbye! Good luck with your studies!\n")
            break

        # --- Add user message to conversation history ---
        messages.append({"role": "user", "content": user_input})

        # --- Call LLM ---
        assistant_msg = call_llm(messages)

        if assistant_msg is None:
            # API error already printed by call_llm(); let the user retry.
            print("Agent: Sorry, I encountered an error. Please try again.\n")
            # Remove the user message so we don't accumulate broken state.
            messages.pop()
            continue

        # ---------------------------------------------------------------
        #  Iterative tool-calling loop
        #  The LLM may request one or more tools. We execute them all,
        #  feed the results back, and call the LLM again — repeating
        #  until the LLM produces a final text answer (no more tool calls).
        # ---------------------------------------------------------------

        while assistant_msg.tool_calls:

            # Save the assistant's tool-request message into history.
            messages.append(assistant_msg)

            for tool_call in assistant_msg.tool_calls:

                tool_name = tool_call.function.name

                # --- Parse arguments (handle malformed JSON) ---
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except (json.JSONDecodeError, TypeError) as exc:
                    if DEBUG_MODE:
                        print(f"\n  [DEBUG] Malformed tool arguments: {exc}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps({
                            "error": f"Malformed JSON arguments: {exc}"
                        })
                    })
                    continue

                # --- Execute the tool (validates internally) ---
                result = execute_tool(tool_name, arguments)

                if DEBUG_MODE:
                    print(f"\n  --- TOOL REQUEST ---")
                    print(f"  Tool:      {tool_name}")
                    print(f"  Arguments: {arguments}")
                    print(f"  --- TOOL RESULT ---")
                    print(f"  {result}")

                # --- Send tool result back to conversation history ---
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })

            # --- Call LLM again with the tool results ---
            assistant_msg = call_llm(messages)

            if assistant_msg is None:
                print("Agent: Sorry, I encountered an error. Please try again.\n")
                break

        # ---------------------------------------------------------------
        #  Final answer — no more tool calls
        # ---------------------------------------------------------------

        if assistant_msg is not None:
            # Save the final assistant message into conversation history.
            messages.append({
                "role": "assistant",
                "content": assistant_msg.content or ""
            })

            print(f"\nAgent: {assistant_msg.content}\n")


# ===========================================================================
#  ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    run_agent()