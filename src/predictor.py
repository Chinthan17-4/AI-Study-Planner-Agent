import joblib
import pandas as pd


MODEL_PATH = "models/study_hours_model.pkl"


def recommend_study_time(
    difficulty,
    previous_score,
    days_left,
    completion_percent
):
    # Load the trained model
    model = joblib.load(MODEL_PATH)

    # Create input data
    student_data = pd.DataFrame([{
        "difficulty": difficulty,
        "previous_score": previous_score,
        "days_left": days_left,
        "completion_percent": completion_percent
    }])

    # Make prediction
    prediction = model.predict(student_data)[0]

    return round(prediction, 2)


if __name__ == "__main__":

    study_hours = recommend_study_time(
        difficulty=5,
        previous_score=45,
        days_left=2,
        completion_percent=30
    )

    print(f"\nRecommended study time: {study_hours} hours")