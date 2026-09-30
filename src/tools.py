from predictor import recommend_study_time
from priority_tool import calculate_priority


def predict_study_time(
    difficulty,
    previous_score,
    days_left,
    completion_percent
):
    """
    Tool 1:
    Predict recommended study time.
    """

    study_hours = recommend_study_time(
        difficulty=difficulty,
        previous_score=previous_score,
        days_left=days_left,
        completion_percent=completion_percent
    )

    return {
        "recommended_hours": float(study_hours)
    }


def prioritize_subject(
    credits,
    days_left,
    previous_marks
):
    """
    Tool 2:
    Calculate subject priority.
    """

    result = calculate_priority(
        credits=credits,
        days_left=days_left,
        previous_marks=previous_marks
    )

    return result