import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# Load the dataset
data = pd.read_csv('Creditcard_data.csv')

# Check class distribution
print("Class distribution before balancing:")
print(data['Class'].value_counts())

# Balance the dataset using SMOTE
X = data.drop('Class', axis=1)
y = data['Class']
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

print("Class distribution after balancing:")
print(pd.Series(y_balanced).value_counts())

# Define sampling techniques
# sampling_techniques = {
#     'Sampling1': RandomUnderSampler(random_state=42),
#     'Sampling2': SMOTE(random_state=42),
#     'Sampling3': SMOTETomek(random_state=42),
#     'Sampling4': RandomUnderSampler(random_state=42, sampling_strategy=0.8),
#     'Sampling5': SMOTE(random_state=42, sampling_strategy=0.8)
# }
sampling_techniques = {
    'Sampling1': RandomUnderSampler(random_state=42),
    'Sampling2': SMOTE(random_state=42),
    'Sampling3': SMOTETomek(random_state=42),
    'Sampling4': RandomUnderSampler(random_state=42, sampling_strategy='auto'),  # Automatically adjusts strategy
    'Sampling5': SMOTE(random_state=42, sampling_strategy='auto')
}

# Define machine learning models
models = {
    'M1': RandomForestClassifier(random_state=42),
    'M2': LogisticRegression(max_iter=1000, random_state=42),
    'M3': SVC(kernel='linear', random_state=42),
    'M4': DecisionTreeClassifier(random_state=42),
    'M5': GaussianNB()
}

# Create samples and evaluate models
results = {}

for sampling_name, sampler in sampling_techniques.items():
    print(f"Applying {sampling_name}...")
    X_sampled, y_sampled = sampler.fit_resample(X_balanced, y_balanced)
    X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.2, random_state=42)

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        if sampling_name not in results:
            results[sampling_name] = {}
        results[sampling_name][model_name] = accuracy

# Display results
results_df = pd.DataFrame(results)
print("\nModel performance across sampling techniques:")
print(results_df)

# Save results to a CSV file
results_df.to_csv('sampling_model_results.csv', index=True)

print("Results saved to 'sampling_model_results.csv'")
