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
            "name": "predict_study_time",
            "description": (
                "Predict the recommended number of study hours "
                "for a student based on subject difficulty, "
                "previous score, days remaining, and completion percentage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "difficulty": {
                        "type": "integer",
                        "description": "Difficulty of the subject from 1 to 5."
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
                        "description": "Percentage of the subject already completed."
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
    You are a study planning assistant.
    Use the available tools whenever their information is needed
    to make a study recommendation.
    """
},
    {
    "role": "user",
    "content": """
    I have an AI exam in 2 days.

    Subject details:
    - Subject: AI
    - Credits: 4
    - Previous score: 45
    - Difficulty: 5 out of 5
    - Syllabus completion: 30%

    Analyze my situation and tell me:
    1. How much priority should I give this subject?
    2. How many hours should I study?
    3. Give me a short study recommendation.

    Use the available tools whenever needed.
    """
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