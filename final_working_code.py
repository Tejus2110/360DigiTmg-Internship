import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = r'D:\COLLEGE\2 year\SECOND_YEAR_INTERNSHIP\360digitmg\projects\Extracted Templates\Machine Downtime.csv'
df_new = pd.read_csv(file_path)

# Changing the values of the 0 to null then finally to the mean value of each column

# Check the number of columns in the DataFrame
num_columns = df_new.shape[1]
print(f"Number of columns in the DataFrame: {num_columns}")

# Define the start and end column indices for 1-based columns 4 to 15
start_col = 3  # Corresponds to column 4 in 1-based indexing
end_col = 14  # Corresponds to column 15 in 1-based indexing

# Ensure we have enough columns
if num_columns < end_col + 1:
    raise ValueError(f"DataFrame has only {num_columns} columns, but at least 15 are required.")

print(f"Processing columns from {start_col + 1} to {end_col + 1} (1-based indexing)")

# Replace zeros with NaN in the specified range
df_new.iloc[:, start_col:end_col + 1] = df_new.iloc[:, start_col:end_col + 1].replace(0, pd.NA)

# Calculate the mean of each column in the specified range
mean_values = df_new.iloc[:, start_col:end_col + 1].mean()
print("Mean values for columns 4 to 15:", mean_values)

# Replace NaN values with the mean value of the respective column
for col in range(start_col, end_col + 1):
    df_new.iloc[:, col] = df_new.iloc[:, col].fillna(mean_values.iloc[col - start_col])

# Print the modified DataFrame
print(df_new)

# If you want to save the modified DataFrame back to a CSV file
# df_new.to_csv('modified_file.csv', index=False)

# ____________ UNIVARIATE ANALYSIS _______________

# Plot the distribution of machine failures
sns.countplot(x='Downtime', data=df_new)
plt.title('Distribution of Machine Failures')
plt.xlabel('Downtime')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['No_Machine Failure', 'Machine_Failure'])
plt.show()

# Sensor data distribution

# List of columns to convert to numeric
columns_to_convert = [
    'Hydraulic_Pressure', 'Coolant_Pressure', 'Air_System_Pressure', 
    'Coolant_Temperature', 'Hydraulic_Oil_Temperature', 
    'Spindle_Bearing_Temperature', 'Spindle_Vibration', 'Tool_Vibration', 
    'Spindle_Speed', 'Voltage', 'Torque', 'Cutting'
]

# Convert specified columns to numeric
for column in columns_to_convert:
    if column in df_new.columns:
        df_new[column] = pd.to_numeric(df_new[column], errors='coerce')

sensor_columns = [
    'Hydraulic_Pressure', 'Coolant_Pressure', 'Air_System_Pressure', 
    'Coolant_Temperature', 'Hydraulic_Oil_Temperature', 
    'Spindle_Bearing_Temperature', 'Spindle_Vibration', 'Tool_Vibration', 
    'Spindle_Speed', 'Voltage', 'Torque', 'Cutting'
]

# Plot sensor data distributions
for column in sensor_columns:
    if column in df_new.columns:
        if pd.api.types.is_numeric_dtype(df_new[column]):
            df_new[column].hist()
            plt.title(f'{column} Data Distribution')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.show()
        else:
            print(f'{column} is not numeric and cannot be plotted.')
    else:
        print(f'{column} not found in DataFrame.')

# ___________________ BIVARIATE ANALYSIS _________________

# SENSOR AND MACHINE FAILURE BOX PLOTS

for column in sensor_columns:
    if column in df_new.columns and pd.api.types.is_numeric_dtype(df_new[column]):
        sns.boxplot(x='Downtime', y=column, data=df_new)
        plt.title(f'{column} vs Machine Failures')
        plt.xlabel('Downtime')
        plt.ylabel(column)
        plt.xticks(ticks=[0, 1], labels=['No_Machine_Failure', 'Machine_Failure'])
        plt.show()
    else:
        print(f'{column} is either not found in the DataFrame or is not numeric and will be skipped.')

# # COOREALTION MATRIX

selected_columns = [
    'Hydraulic_Pressure', 'Coolant_Pressure', 'Air_System_Pressure', 
    'Coolant_Temperature', 'Hydraulic_Oil_Temperature', 
    'Spindle_Bearing_Temperature', 'Spindle_Vibration', 'Tool_Vibration', 
    'Spindle_Speed', 'Voltage', 'Torque', 'Cutting'
]

df_selected = df_new[selected_columns]
corr_selected = df_selected.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_selected, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix for Selected Columns')
plt.show()


# FINDING CO-RELATION BETWEEN MULTIPLE COLUMNS AND THE MACHINE DOWNTIME COLUMN
df_new['Downtime_Encoded'] = df_new['Downtime'].map({'Machine_Failure': 1, 'No_Machine_Failure': 0})

# Calculate correlations between each sensor column and Downtime_Encoded
correlations = {}
for column in sensor_columns:
    if column in df_new.columns:
        correlations[column] = df_new[column].corr(df_new['Downtime_Encoded'])

# Convert the correlations dictionary to a DataFrame
corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation_with_Downtime'])

# Print the correlation DataFrame to see the correlation values
print(corr_df)

# Visualize the correlations
plt.figure(figsize=(10, 8))
sns.heatmap(corr_df, annot=True, cmap='coolwarm', linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation of Sensor Data with Downtime')
plt.show()

# Time Series Analysis

# Counting machine failures across months and then plotting it on month basis to see the trend of machine failure
df_new['Date'] = pd.to_datetime(df_new['Date'], format='%d-%m-%Y')
df_new.set_index('Date')['Downtime'].resample('M').apply(lambda x: (x == 'Machine_Failure').sum()).plot()
plt.title('Monthly Machine Failures')
plt.xlabel('Date')
plt.ylabel('Number of Failures')
plt.show()

# Rolling Mean And Variance
for column in sensor_columns:
    if column in df_new.columns:
        df_new[f'{column}_rolling_mean'] = df_new[column].rolling(window=30).mean()
        df_new[f'{column}_rolling_var'] = df_new[column].rolling(window=30).var()
        df_new[[f'{column}_rolling_mean', f'{column}_rolling_var']].plot(subplots=True)
        plt.title(f'30-Day Rolling Mean and Variance for {column}')
        plt.show()

# NORMAL CORELATION CHART

# Assuming corr_df is the DataFrame containing correlation values
# Find the column with the highest correlation

max_corr_column = corr_df.idxmax()[0]
max_corr_value = corr_df.loc[max_corr_column, 'Correlation_with_Downtime']

# Plotting
plt.figure(figsize=(10, 6))  # Adjust figure size if needed

# Plot bar graph
bars = plt.bar(corr_df.index, corr_df['Correlation_with_Downtime'], color='skyblue')

# Highlight the bar with the highest correlation
for bar in bars:
    if bar.get_height() == max_corr_value:
        bar.set_color('orange')  # Change color of the bar with the highest correlation

# Customize plot
plt.title('Correlation with Machine Downtime')
plt.xlabel('Sensor Columns')
plt.ylabel('Correlation')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.grid(axis='y')  # Add grid for y-axis
plt.show()

