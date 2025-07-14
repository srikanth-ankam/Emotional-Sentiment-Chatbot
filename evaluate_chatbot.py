import pandas as pd
import sys
import os

# --- CRITICAL: Ensure your 'response_logic.py' is accessible ---
# This line adds the parent directory of the current script to Python's search path.
# This is usually where 'response_logic.py' would be if your structure is:
# your_project/
# ├── response_logic.py
# └── evaluation_script_directory/
#     └── evaluate_chatbot.py
#
# If 'response_logic.py' is in the SAME directory as 'evaluate_chatbot.py',
# you might not need the os.path.join(..., "..") part, or simply remove sys.path.append.
# Adjust this path based on your actual file structure.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    # This imports the actual get_emotion from your response_logic.py
    from response_logic import get_emotion
    print("Successfully imported 'get_emotion' from 'response_logic.py'.")
except ImportError:
    print("ERROR: Could not import 'get_emotion' from 'response_logic.py'.")
    print("Please ensure:")
    print("1. 'response_logic.py' exists in the expected location (e.g., in the parent directory or the same directory).")
    print("2. The 'sys.path.append' line is correctly set for your file structure.")
    print("3. 'response_logic.py' does not have errors preventing its import (e.g., missing 'transformers' library or 'fine_tuned_model').")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    print("Please check your response_logic.py and ensure all dependencies are installed.")
    sys.exit(1)


# Define the path to your CSV test file
# Assuming 'emotion_dataset_expanded.csv' is in the same directory as this script,
# or provide the full path if it's elsewhere.
csv_file_path = "extended.csv"

# Load test cases from CSV using pandas
try:
    test_cases_df = pd.read_csv(csv_file_path)
    print(f"CSV file '{csv_file_path}' loaded successfully.")
    print("\nFirst 5 rows of the dataset:")
    print(test_cases_df.head())
    print("\nColumn information:")
    print(test_cases_df.info())
except FileNotFoundError:
    print(f"ERROR: The CSV file '{csv_file_path}' was not found.")
    print("Please make sure the file exists in the same directory as this script, or provide the full path.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred while reading the CSV file: {e}")
    sys.exit(1)

# Assuming column names 'text' for input and 'emotion' for expected emotion
# If your CSV has different column names, please update these variables:
input_col = 'text'
expected_emotion_col = 'emotion'

# Validate required columns
if input_col not in test_cases_df.columns or expected_emotion_col not in test_cases_df.columns:
    print(f"ERROR: The CSV file must contain '{input_col}' and '{expected_emotion_col}' columns for evaluation.")
    print(f"Available columns are: {test_cases_df.columns.tolist()}")
    print(f"Please ensure your CSV has columns named '{input_col}' for input and '{expected_emotion_col}' for expected labels, or update 'input_col' and 'expected_emotion_col' variables in this script.")
    sys.exit(1)

# Evaluate chatbot predictions
results = []
correct_predictions = 0
total_predictions = len(test_cases_df)

print(f"\n--- Starting Chatbot Evaluation with {total_predictions} test cases ---")
print("NOTE: This evaluation is now using your 'get_emotion' function from 'response_logic.py'.")
print("      Ensure your 'fine_tuned_model' is correctly accessible by 'response_logic.py'.")
print("-" * 60)

for index, row in test_cases_df.iterrows():
    user_input = str(row[input_col]) # Ensure input is string
    expected_emotion = str(row[expected_emotion_col]) # Ensure expected is string

    # Use the actual get_emotion from your response_logic
    predicted_emotion = get_emotion(user_input)
    is_correct = predicted_emotion.lower() == expected_emotion.lower()

    results.append({
        "input": user_input,
        "expected": expected_emotion,
        "predicted": predicted_emotion,
        "correct": is_correct
    })

    if is_correct:
        correct_predictions += 1

# Print detailed results (showing only first 10 examples to keep output manageable)
print("\n--- Detailed Evaluation Results (first 10 examples) ---")
for i, result in enumerate(results):
    if i >= 10:
        break
    status_icon = "✅ Correct" if result['correct'] else "❌ Incorrect"
    print(f"Input:    {result['input']}")
    print(f"Expected: {result['expected'].capitalize()} | Predicted: {result['predicted'].capitalize()} | {status_icon}")
    print("-" * 60)

# Calculate and print accuracy
accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0

print("\n--- Chatbot Evaluation Summary ---\n")
print(f"Total Test Cases: {total_predictions}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Overall Accuracy: {accuracy:.2%}") # Formats as percentage with 2 decimal places
print("\n--- Evaluation Complete ---")