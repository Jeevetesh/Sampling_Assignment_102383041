# Sampling Assignment

This repository contains the implementation of various sampling techniques to balance a dataset and evaluate machine learning models. The project uses a credit card dataset to analyze which sampling technique provides the highest accuracy for different models.

## Files

1. **Creditcard_data.csv**: The dataset used for the assignment.
2. **sampling_model_results.csv**: The output file containing the results of model accuracy for different sampling techniques.
3. **Sampling_Assignment.py**: The main Python script that implements the sampling techniques and evaluates the models.
4. **sampling_summary.py**: The Python script to summarize the best sampling techniques and highest accuracies for each model.
5. **README.md**: This file.

## Steps to Execute

1. **Install Required Libraries**:
   Ensure you have the necessary libraries installed:
   ```bash
   pip install pandas scikit-learn imbalanced-learn
   ```

2. **Run the Main Script**:
   Execute the script `Sampling_Assignment.py` to perform the following tasks:
   - Load the dataset.
   - Balance the dataset using SMOTE.
   - Apply five different sampling techniques.
   - Train five machine learning models.
   - Calculate and save the accuracy results.

   ```bash
   python Sampling_Assignment.py
   ```

3. **Summarize Results**:
   Execute `sampling_summary.py` to:
   - Load the `sampling_model_results.csv`.
   - Identify the best sampling technique for each model.
   - Determine the overall best sampling technique across all models.
   - Save the summary to a new CSV file (`best_sampling_summary.csv`).

   ```bash
   python sampling_summary.py
   ```

4. **Analyze the Results**:
   - View `sampling_model_results.csv` for detailed accuracy scores.
   - View `best_sampling_summary.csv` for the summary of the best sampling techniques and highest accuracies.

## Results Summary

- The best sampling technique for each model is stored in `best_sampling_summary.csv`.
- The highest overall accuracy across all models is achieved using **Sampling3**, with an average accuracy of **94.30%**.

## Key Functions and Libraries Used

- **pandas**: For data manipulation and analysis.
- **scikit-learn**: For implementing machine learning models and evaluation.
- **imbalanced-learn**: For handling imbalanced datasets with sampling techniques such as SMOTE and RandomUnderSampler.

## Notes

- Ensure that the dataset file `Creditcard_data.csv` is in the same directory as the scripts.
- Modify the paths in the scripts if necessary to match your file structure.

## Contact

If you have any questions or need further assistance, feel free to reach out.
