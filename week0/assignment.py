import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
import psutil
import uuid

# Suppress warnings
warnings.filterwarnings("ignore")


# Function to handle missing data (NA or empty values)
def handle_missing_data(data_frame):
    """
    Drops rows with missing values (NA or empty) from the given DataFrame.

    Args:
        data_frame (pandas.DataFrame): The DataFrame to process.

    Returns:
        pandas.DataFrame: The DataFrame with missing values removed.
    """

    data_frame.dropna(inplace=True)
    return data_frame


# Function to calculate descriptive statistics (mean, mode, median) for a column
def calculate_descriptive_statistics(data_frame, column_name):
    """
    Calculates mean, mode, and median for a specified column in the DataFrame.

    Args:
        data_frame (pandas.DataFrame): The DataFrame to process.
        column_name (str): The name of the column for which to calculate statistics.

    Returns:
        tuple: A tuple containing mean, mode, and median values.
    """

    mean = data_frame[column_name].mean()
    mode = data_frame[column_name].mode().values[0]
    median = data_frame[column_name].median()
    return mean, mode, median


# Function to create a histogram for visualizing the distribution of a column
def visualize_data_distribution(data_frame, column_name):
    """
    Creates a histogram to visualize the distribution of a column in the DataFrame.

    Args:
        data_frame (pandas.DataFrame): The DataFrame to process.
        column_name (str): The name of the column to visualize.
    """

    plt.figure(figsize=(10, 6))
    sns.histplot(data_frame[column_name], kde=True)  # Include KDE for smoother distribution
    plt.title(f"Distribution of {column_name}")
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.show()


# Process 'athlete_events.csv' data
start_time = time.time()
memory_before = psutil.Process().memory_info().rss / (1024 * 1024)

# Read the CSV data into a DataFrame
athlete_events_df = pd.read_csv("Week-0 Assignment/Week-0 Assignment/athlete_events.csv")
sales_data_sample_df = pd.read_csv(
    "Week-0 Assignment/Week-0 Assignment/sales_data_sample.csv", encoding="Windows-1252"
)

# Handle missing data in the athlete events DataFrame
athlete_events_df = handle_missing_data(athlete_events_df)

# Assuming 'Age' is the column of interest
column_to_analyze = "Age"
mean, mode, median = calculate_descriptive_statistics(athlete_events_df, column_to_analyze)
print(f"Mean {column_to_analyze}: {mean}")
print(f"Mode {column_to_analyze}: {mode}")
print(f"Median {column_to_analyze}: {median}")

# Create a histogram to visualize the age distribution
visualize_data_distribution(athlete_events_df, column_to_analyze)

# Process 'sales_data_sample.csv' data
sales_data_sample_df = handle_missing_data(sales_data_sample_df)

# Assuming 'SALES' is the column of interest
column_to_analyze = "SALES"
mean, mode, median = calculate_descriptive_statistics(sales_data_sample_df, column_to_analyze)
print(f"\nMean {column_to_analyze}: {mean}")
print(f"Mode {column_to_analyze}: {mode}")
print(f"Median {column_to_analyze}: {median}")

# Create a histogram to visualize the sales distribution
visualize_data_distribution(sales_data_sample_df, column_to_analyze)

memory_after = psutil.Process().memory_info().rss / (1024 * 1024)
end_time = time.time()

# Get MAC address
mac_address = ":".join(
    ["{:02x}".format((uuid.getnode() >> elements) & 0xff) for elements
