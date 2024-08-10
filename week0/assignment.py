import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import psutil
import uuid
warnings.filterwarnings("ignore")
# Function to handle missing and NULL values
def handle_missing_values(df):
    df.dropna(inplace=True)  # Remove rows with missing values
  
# Function to calculate mean, mode, and median
def calculate_stats(df, attribute):
    mean = df[attribute].mean()
    mode = df[attribute].mode().values[0]
    median = df[attribute].median()
    return mean, mode, median

# Function to visualize a column attribute with histogram
def visualize_attribute(df, attribute):
    plt.figure(figsize=(10, 6))
    sns.histplot(df[attribute], kde=True)  # Include KDE for smoother distribution
    plt.title(f"Distribution of {attribute}")
    plt.xlabel(attribute)
    plt.ylabel("Frequency")
    plt.show()

# Process athlete_events.csv
mac = uuid.getnode()
start_time = time.time()
memory_before = psutil.Process().memory_info().rss / (1024 * 1024)
athlete_events_df = pd.read_csv(r"Week-0 Assignment\Week-0 Assignment\athlete_events.csv")
sales_data_sample_df = pd.read_csv(r"Week-0 Assignment\Week-0 Assignment\sales_data_sample.csv",encoding='Windows-1252')

handle_missing_values(athlete_events_df)

# Assuming 'Age' is the attribute of interest 
attribute = 'Age'
mean, mode, median = calculate_stats(athlete_events_df, attribute)
print(f"Mean {attribute}: {mean}")
print(f"Mode {attribute}: {mode}")
print(f"Median {attribute}: {median}")

visualize_attribute(athlete_events_df, attribute)

# Process sales_data_sample.csv
handle_missing_values(sales_data_sample_df)

# Assuming 'Sales' is the attribute of interest
attribute = 'SALES'
mean, mode, median = calculate_stats(sales_data_sample_df, attribute)
print(f"\nMean {attribute}: {mean}")
print(f"Mode {attribute}: {mode}")
print(f"Median {attribute}: {median}")

visualize_attribute(sales_data_sample_df, attribute)
memory_after = psutil.Process().memory_info().rss / (1024 * 1024)
end_time = time.time()
mac_address = ':'.join(['{:02x}'.format((mac >> elements) & 0xff) for elements in range(0, 8*6, 8)][::-1])
print(f"MAC Address: {mac_address}")
print(f"Memory used: {memory_after - memory_before:.2f} MB")
print(f"Time taken: {end_time - start_time:.4f} seconds")
