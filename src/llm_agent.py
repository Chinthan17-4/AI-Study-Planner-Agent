import os
import json

from dotenv import load_dotenv
from groq import Groq
from tools import prioritize_subject, predict_study_time

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
},
{
    "type": "function",
    "function": {
        "name": "predict_study_time",
        "description": "Predict the recommended number of study hours using the trained ML model.",
        "parameters": {
            "type": "object",
            "properties": {
                "difficulty": {
                    "type": "integer",
                    "description": "Subject difficulty from 1 to 5."
                },
                "previous_score": {
                    "type": "number",
                    "description": "Student's previous score."
                },
                "days_left": {
                    "type": "integer",
                    "description": "Number of days remaining before the exam."
                },
                "completion_percent": {
                    "type": "number",
                    "description": "Percentage of syllabus already completed."
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
}
    
]


messages = [
   {
    "role": "system",
    "content": """
    You are an AI study planning agent.

    You have access to tools that provide factual calculations.

    When the user asks for subject priority, you MUST use
    the prioritize_subject tool.

    When the user asks for recommended study time, you MUST use
    the predict_study_time tool.

    Do not calculate or guess these values yourself when the
    corresponding tool is available.

    After receiving the tool results, use those results to create
    the final study recommendation.
    """
},
    {
    "role": "user",
    "content": """
    Analyze my AI exam situation.

    Subject: AI
    Credits: 4
    Previous score: 45
    Days left: 2
    Difficulty: 5 out of 5
    Syllabus completion: 30%

    I want to know:
    1. The priority of this subject.
    2. How many hours I should study.
    3. A short recommendation based on the tool results.

    Use the appropriate tools to calculate the priority and
    recommended study time.
    """
}
]

print("\n--- AVAILABLE TOOLS ---")

for tool in tools:
    print(tool["function"]["name"])
    
response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)


message = response.choices[0].message

print("\n--- LLM RESPONSE ---")
print(message)

while message.tool_calls:

    # Add the LLM's tool request to the conversation
    messages.append(message)

    for tool_call in message.tool_calls:

        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        print("\n--- TOOL REQUEST ---")
        print("Tool:", tool_name)
        print("Arguments:", arguments)

        if tool_name == "prioritize_subject":

            result = prioritize_subject(
                credits=arguments["credits"],
                days_left=arguments["days_left"],
                previous_marks=arguments["previous_marks"]
            )

        elif tool_name == "predict_study_time":

            result = predict_study_time(
                difficulty=arguments["difficulty"],
                previous_score=arguments["previous_score"],
                days_left=arguments["days_left"],
                completion_percent=arguments["completion_percent"]
            )

        else:

            result = {
                "error": f"Unknown tool: {tool_name}"
            }

        print("\n--- TOOL RESULT ---")
        print(result)

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        })

    # Ask the LLM what to do next
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

    message = response.choices[0].message


print("\n--- FINAL LLM RESPONSE ---")
print(message.content)