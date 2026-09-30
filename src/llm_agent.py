import os
import json

from dotenv import load_dotenv
from groq import Groq

from tools import prioritize_subject


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY is not configured.")

client = Groq(api_key=api_key)


tools = [
    {
        "type": "function",
        "function": {
            "name": "prioritize_subject",
            "description": "Calculate the study priority of a subject using credits, days remaining, and previous marks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "credits": {
                        "type": "integer",
                        "description": "Number of academic credits for the subject."
                    },
                    "days_left": {
                        "type": "integer",
                        "description": "Number of days remaining before the exam."
                    },
                    "previous_marks": {
                        "type": "number",
                        "description": "Student's previous marks for the subject."
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


messages = [
    {
        "role": "system",
        "content": (
            "You are a study planning assistant. "
            "Use the available tool when you need to determine "
            "a subject's study priority."
        )
    },
    {
        "role": "user",
        "content": (
            "I have an AI exam in 2 days. "
            "The subject has 4 credits and I previously scored 45 marks. "
            "Determine its priority."
        )
    }
]


response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)


message = response.choices[0].message

print("\n--- LLM RESPONSE ---")
print(message)

if message.tool_calls:

    tool_call = message.tool_calls[0]

    print("\n--- TOOL REQUEST ---")
    print("Tool:", tool_call.function.name)
    print("Arguments:", tool_call.function.arguments)

    arguments = json.loads(tool_call.function.arguments)

    result = prioritize_subject(
        credits=arguments["credits"],
        days_left=arguments["days_left"],
        previous_marks=arguments["previous_marks"]
    )

    print("\n--- TOOL RESULT ---")
    print(result)

    # Add the assistant's tool request to the conversation
    messages.append(message)

    # Add the tool result to the conversation
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result)
    })

    # Ask the LLM to produce the final answer
    final_response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        tools=tools
    )

    print("\n--- FINAL LLM RESPONSE ---")
    print(final_response.choices[0].message.content)