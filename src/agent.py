from tools import predict_study_time, prioritize_subject


def study_agent(student):

    print("\n--- STUDY AGENT ---")

    print("\nStudent information:")
    print(student)

    # STEP 1: Agent checks subject priority
    print("\n--- AGENT DECISION ---")
    print("First, I need to determine subject priority.")

    priority_result = prioritize_subject(
        credits=student["credits"],
        days_left=student["days_left"],
        previous_marks=student["previous_score"]
    )

    print("Priority result:")
    print(priority_result)

    # STEP 2: Agent decides whether study-time prediction is needed
    if priority_result["priority"] == "HIGH":

        print("\nDecision:")
        print("Priority is HIGH, so I need a study-time prediction.")

        study_result = predict_study_time(
            difficulty=student["difficulty"],
            previous_score=student["previous_score"],
            days_left=student["days_left"],
            completion_percent=student["completion_percent"]
        )

    else:

        print("\nDecision:")
        print("Priority is not HIGH, so I will use a lower study allocation.")

        study_result = {
            "recommended_hours": 2.0
        }

    print("\nStudy-time result:")
    print(study_result)

    return {
        "priority": priority_result,
        "study_time": study_result
    }


if __name__ == "__main__":

    student = {
        "subject": "AI",
        "credits": 4,
        "difficulty": 5,
        "previous_score": 45,
        "days_left": 2,
        "completion_percent": 30
    }

    study_agent(student)