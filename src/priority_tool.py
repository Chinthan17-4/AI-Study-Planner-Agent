def calculate_priority(credits, days_left, previous_marks):

    # 1. Higher credits = higher priority
    credit_score = credits / 5

    # 2. Fewer days left = higher urgency
    urgency_score = (30 - days_left) / 29
    # 3. Lower marks = higher weakness
    weakness_score = (100 - previous_marks) / 100

    # 4. Combine the three factors
    priority_score = (
        credit_score * 0.25
        + urgency_score * 0.40
        + weakness_score * 0.35
    )

    # 5. Convert score into priority level
    if priority_score >= 0.70:
        priority = "HIGH"
    elif priority_score >= 0.40:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    return {
        "priority_score": round(priority_score, 2),
        "priority": priority
    }


if __name__ == "__main__":

    result = calculate_priority(
        credits=4,
        days_left=2,
        previous_marks=45
    )

    print(result)