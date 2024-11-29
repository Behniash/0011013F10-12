import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import resample

#nom 1:

data=pd.read_csv('D:/bank-additional-full.csv') 
df = pd.DataFrame(data)
print(df)


df.info()
df.head()
df.describe()

col=list(df.columns)
print(col)

cd = pd.DataFrame(col)
print(cd)

print("----------------------------------------------------------------------------")
print("----------------------------------------------------------------------------")

#nom 2:

duplicateds={}

for column in df.columns:
    duplicated_values = df[column][df[column].duplicated()].unique()
    duplicateds[column] = duplicated_values.tolist()

for col, values in duplicateds.items():
    print(f"column '{col}'has duplicateds: '{values}'")

print("----------------------------------------------------------------------------")
print("----------------------------------------------------------------------------")

#nom 3:

def find_categorical_columns(df):
    categorical_columns = []
    for column in df.columns:
        unique_count = df[column].nunique()
        total_count = df[column].count()
        if unique_count / total_count < 0.5:
            categorical_columns.append(column)
    return categorical_columns

categorical_columns = find_categorical_columns(df)
print("Categorical columns:", categorical_columns)

print("----------------------------------------------------------------------------")
print("----------------------------------------------------------------------------")

#nom 4:


def normalize_only_numeric_columns(df):
    numeric_columns = df.select_dtypes(include=['number']).columns
    
    normalized_df = df.copy()
    
    for column in numeric_columns:
        min_value = df[column].min()
        max_value = df[column].max()
        if max_value - min_value != 0:
            normalized_df[column] = (df[column] - min_value) / (max_value - min_value)
        else:
            normalized_df[column] = 0 
    
    return normalized_df

normalized_data_info = normalize_only_numeric_columns(df)
print("Normalized Data (Only Numeric Columns):")
print(normalized_data_info)

print("----------------------------------------------------------------------------")
print("----------------------------------------------------------------------------")

#nom 5:

numerical_features = data.select_dtypes(include=['float64', 'int64']).columns

num_features_count = len(numerical_features)
rows = (num_features_count // 3) + (num_features_count % 3 > 0)
fig, axes = plt.subplots(rows, 3, figsize=(15, rows * 5))
axes = axes.flatten()  

for i, feature in enumerate(numerical_features):
    sns.histplot(data[feature], kde=True, bins=30, color='blue', ax=axes[i])
    axes[i].set_title(f'Distribution of {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')

for j in range(len(numerical_features), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

#nom 6:

numerical_features = data.select_dtypes(include=['float64', 'int64']).columns

def remove_outliers_with_report(df, feature, threshold=1.5):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    filtered_df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    return filtered_df

data_cleaned = data.copy()
for feature in numerical_features:
    if data_cleaned[feature].nunique() > 1:  
        data_cleaned = remove_outliers_with_report(data_cleaned, feature)

num_features_count = len(numerical_features)
rows = (num_features_count * 2 // 3) + (num_features_count * 2 % 3 > 0)  

fig, axes = plt.subplots(rows, 3, figsize=(18, rows * 5))
axes = axes.flatten()  

plot_index = 0
for feature in numerical_features:
    if data[feature].nunique() > 1: 
        sns.boxplot(data=data[feature], ax=axes[plot_index], color="red")
        axes[plot_index].set_title(f"Before Outlier Removal: {feature}")
        axes[plot_index].set_xlabel("")
        plot_index += 1

        sns.boxplot(data=data_cleaned[feature], ax=axes[plot_index], color="green")
        axes[plot_index].set_title(f"After Outlier Removal: {feature}")
        axes[plot_index].set_xlabel("")
        plot_index += 1

for j in range(plot_index, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

data_cleaned.to_csv('cleaned_data_with_boxplots.csv', index=False)

#nom-8


# Step 2: Specify the target column (class labels)
target_column = 'y'  # Replace 'y' with your target column name

# Step 3: Count samples in each class
class_counts = data[target_column].value_counts()
class_percentages = data[target_column].value_counts(normalize=True) * 100

# Display class counts and percentages
print("Class Distribution:")
print(class_counts)
print("\nClass Percentages:")
print(class_percentages)

# Step 4: Visualize class distribution
plt.figure(figsize=(12, 6))

# Bar plot for class counts
plt.subplot(1, 2, 1)
sns.barplot(x=class_counts.index, y=class_counts.values, palette="viridis")
plt.title("Class Counts")
plt.xlabel("Classes")
plt.ylabel("Number of Samples")

# Pie chart for class percentages
plt.subplot(1, 2, 2)
plt.pie(class_percentages.values, labels=class_percentages.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("viridis", len(class_percentages)))
plt.title("Class Percentages")

plt.tight_layout()
plt.show()

# Step 5: Check for imbalance
threshold = 20  # Adjust the threshold for imbalance (e.g., 20%)
max_percentage = class_percentages.max()
min_percentage = class_percentages.min()

if max_percentage - min_percentage > threshold:
    print("\nThe dataset is imbalanced.")
else:
    print("\nThe dataset is balanced.")


#nom 9:


# Step 2: Specify the target column (class labels)
target_column = 'y'  # Replace 'y' with your target column name

# Step 3: Separate majority and minority classes
class_counts = data[target_column].value_counts()
print("Class Counts Before Undersampling:")
print(class_counts)

majority_class = class_counts.idxmax()
minority_class = class_counts.idxmin()

majority_data = data[data[target_column] == majority_class]
minority_data = data[data[target_column] == minority_class]

# Step 4: Downsample majority class
majority_downsampled = resample(majority_data,
                                replace=False,  # Sample without replacement
                                n_samples=len(minority_data),  # Match minority class size
                                random_state=42)  # For reproducibility

# Step 5: Combine minority class with downsampled majority class
balanced_data = pd.concat([majority_downsampled, minority_data])

# Step 6: Display the results
print("\nClass Counts After Undersampling:")
print(balanced_data[target_column].value_counts())

# Save balanced data to a new file (optional)
balanced_data.to_csv('balanced_data_undersampling.csv', index=False)