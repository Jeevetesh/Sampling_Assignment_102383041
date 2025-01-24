import pandas as pd

# Load the uploaded results CSV file
file_path = 'sampling_model_results.csv'  # Replace with the correct path
results_df = pd.read_csv(file_path, index_col=0)

# Determine the sampling technique with the highest accuracy for each model
best_sampling_for_models = results_df.idxmax(axis=1)
highest_accuracies = results_df.max(axis=1)

# Combine the best sampling technique with its corresponding accuracy
best_sampling_summary = pd.DataFrame({
    'Best Sampling Technique': best_sampling_for_models,
    'Highest Accuracy': highest_accuracies
})

# Display the summary
print(best_sampling_summary)

# Save the summary to a CSV file
best_sampling_summary.to_csv('best_sampling_summary.csv', index=True)
print("Best sampling summary saved to 'best_sampling_summary.csv'")

# Find the sampling technique with the highest overall average accuracy
average_accuracies = results_df.mean(axis=0)
best_overall_sampling_technique = average_accuracies.idxmax()
highest_overall_accuracy = average_accuracies.max()

print(f"Best overall sampling technique: {best_overall_sampling_technique}")
print(f"Highest average accuracy: {highest_overall_accuracy:.2f}")
