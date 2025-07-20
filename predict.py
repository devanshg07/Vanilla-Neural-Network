import dill

# Load the trained model
with open('best_model.pkl', 'rb') as f:
    nn = dill.load(f)

# Example: Predict GPA for a new student
# Input order: Age, StudyTimeWeekly, Absences, TotalActivities
# Example input: 17 years old, 12 hours/week, 3 absences, 2 activities
input_features = [17, 26, 23, 3]

# The model expects a list of floats
predicted_gpa = nn(input_features)
print("Raw model output:", predicted_gpa)

print(f"Predicted GPA: {predicted_gpa.data:.2f}") 