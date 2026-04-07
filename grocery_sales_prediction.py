#%%#
# Author: Bansi Kamani
### Step 1
######################################################################
### Import essential library and load the data
## Import pandas, numpy, and etc.
import numpy as np
import pandas as pd
import seaborn as sns
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

## Read the data and understanding our data set, assign your personal file path for the purpose of loading data
product = pd.read_csv("Products_Information.csv")

## Using info() and head() functions to briefly observe the overall pattern of the data
product_info = product.info()
product_describe = product.describe()
product_rows = product.head()

## Check the data set
print(product_info)
print("------------------------------------------------")
print(" ")
print(product_describe)
print("------------------------------------------------")
print(" ")
print(product_rows)

###-------------------------------------------------------------------------------
#%%#
### Step 2
######################################################################
### Data pre-processing, such as handle Missing Values and etc.

## Deal with missing value, firstly check the missing value
missing_values = product.isnull().sum()

## Convert the 'date' column to datetime format and set it as the index
product['date'] = pd.to_datetime(product['date'])
product.set_index('date', inplace=True)

## Check the latest processing result
print(missing_values)
print("------------------------------------------------")
print(" ")
print(product.head())

## Save processed data
# product.to_csv("processed_product_data.csv")

###-------------------------------------------------------------------------------
#%%#
### Step 3
######################################################################
### Overall EDA and Graph plotting

## Setting plotting configuration for plots
sns.set(style="darkgrid")
sns.set_theme()
sns.set_context("talk", font_scale=0.8, rc={"lines.linewidth": 1})

## Plot sales over all time (supplementary plot)
overall_sales = product.groupby('date')['sales'].sum()

## Creating the plot
plt.figure(figsize=(14, 7))
plt.plot(overall_sales, label='Total Sales')
plt.title('Overall sales Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.legend()
plt.show()

## Summarize the stats of sales
print(product['sales'].describe())

## Distribution of sales
plt.figure(figsize=(14, 7))
sns.histplot(product['sales'], bins=50, kde=True)
plt.title('Distribution of Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.xlim(0, 15000)
plt.show()

## Sales by Store (important)
plt.figure(figsize=(14, 7))
sns.boxplot(x='store_nbr', y='sales', data=product)
plt.title('Sales by Store')
plt.xlabel('Store Number')
plt.ylabel('Sales')
plt.show()

## Sales by Product (important)
plt.figure(figsize=(14, 7))
sns.boxplot(x='product_type', y='sales', data=product)
plt.title('Sales by Product')
plt.xlabel('Product Type')
plt.ylabel('Sales')
plt.xticks(rotation=90)
plt.show()

###-------------------------------------------------------------------------------
#%%#
### Step 4
######################################################################
### Data advanced processing, group by both store and product (store-product combinations)

## Ensure 'product_type' is of categorical data type
product['product_type'] = product['product_type'].astype('category')

############################################################################
##### Creating a dictionary to hold the data for each product-store combination (Important Step)
############################################################################
segmented_data = {}

## Grouping the data by store and product, and storing each group in the dictionary
for (store, product_type), group in product.groupby(['store_nbr', 'product_type'], observed=True):
    segmented_data[(store, product_type)] = group[['sales', 'special_offer', 'id']]

## Displaying the number of segments created and a sample segment
num_segments = len(segmented_data)
sample_segment_key = list(segmented_data.keys())[1]

## sample_segment_key = list(segmented_data.keys())[0] -> will be store 1's AUTOMOTIVE
sample_segment_data = segmented_data[sample_segment_key].head()

print("Number of segments:", num_segments)
print("------------------------------------------------")
print(" ")
print("Sample segment key:", sample_segment_key)
print("------------------------------------------------")
print(" ")
print(sample_segment_data)

###-------------------------------------------------------------------------------
#%%#
### Step 5
######################################################################
### Individual store and product demonstration (store-product combinations)

## After grouping the data by both store and product, we can select the specific store's product
## Define the store number and product type you are interested in
store_number = 30
product_type = 'BEAUTY'

## Access the sales data for the specific store and product
specific_segment = segmented_data[(store_number, product_type)]

## Display the data using first 15 days data
print("Number of Store:", store_number)
print("Specific Product Type:", product_type)
print("------------------------------------------------")
print(" ")
print(specific_segment.head(15))

###-------------------------------------------------------------------------------
#%%#
### Step 6-1
####################################################################
### Individual combinations of store and product grouping by checking the scales of the sales
### data for the purpose of segmenting the data into zero sales and non-zero sales.

## Set up graph configuration
sns.set(style="darkgrid")
sns.set_theme()
sns.set_context("talk", font_scale=0.8, rc={"lines.linewidth": 1})

## Demonstrate the sales data for store 1 and its 'BABY CARE' product (zero sales throughout)
store_number = 1
product_type = 'BABY CARE'
specific_segment = segmented_data[(store_number, product_type)]

## Ensure 'date' is the index and in the correct format
specific_segment.index = pd.to_datetime(specific_segment.index)

## Filter the data to include only dates up to 2017-07-30
specific_segment = specific_segment[:"2017-07-30"]

## Creating the plot
plt.figure(figsize=(14, 7))
plt.plot(specific_segment.index, specific_segment['sales'], linewidth=1)
plt.title(f'Sales Over Time for Store {store_number} - {product_type}')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

## Demonstrate the sales data for store 1 and its 'BEVERAGES' product (non-zero sales)
store_number = 1
product_type = 'BEVERAGES'
specific_segment = segmented_data[(store_number, product_type)]

## Ensure 'date' is the index and in the correct format
specific_segment.index = pd.to_datetime(specific_segment.index)

## Filter the data to include only dates up to 2017-07-30
specific_segment = specific_segment[:"2017-07-30"]

## Creating the plot
plt.figure(figsize=(22, 6))
plt.plot(specific_segment.index, specific_segment['sales'], linewidth=1)
plt.title(f'Sales Over Time for Store {store_number} - {product_type}')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

## Demonstrate the sales data for store 1 and its 'BOOKS' product (mixed zero and non-zero sales)
store_number = 1
product_type = 'BOOKS'
specific_segment = segmented_data[(store_number, product_type)]

## Ensure 'date' is the index and in the correct format
specific_segment.index = pd.to_datetime(specific_segment.index)

## Filter the data to include only dates up to 2017-07-30
specific_segment = specific_segment[:"2017-07-30"]

## Creating the plot
plt.figure(figsize=(22, 6))
plt.plot(specific_segment.index, specific_segment['sales'], linewidth=1)
plt.title(f'Sales Over Time for Store {store_number} - {product_type}')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.show()

### Conclusion ###
### The sales data for store 1 - 'BABY CARE' consists of zero sales throughout the entire period.
### The sales data for store 1 - 'BEVERAGES' consists of non-zero sales throughout the entire period.
### Therefore, the data can be segmented into zero sales and non-zero sales groups.
### However, store 1 - 'BOOKS' shows both zero and non-zero sales over the period.
### In further steps (in the forecaster class), a function will be built to filter out leading
### zero values when a combination shows initial zero sales but transitions to non-zero sales later.

###-------------------------------------------------------------------------------
#%%#
### Step 6-2
####################################################################
### Create several functions to segment the data into zero sales and non-zero sales

## Function to load and process data
def load_and_process_data(filepath):
    product = pd.read_csv(filepath)
    product['date'] = pd.to_datetime(product['date'])
    product.set_index('date', inplace=True)
    product['product_type'] = product['product_type'].astype('category')
    return product

## Function to segment data
def segment_data(product):
    segmented_data = {}
    for (store, product_type), group in product.groupby(['store_nbr', 'product_type'], observed=True):
        segmented_data[(store, product_type)] = group
    return segmented_data

## Function to divide segments into zero and non-zero sales
def divide_segments(segmented_data):
    zero_sales_segments = []
    non_zero_sales_segments = []

    for key, segment in segmented_data.items():
        store, product_type = key

        ## Filter data up to 2017-07-30
        segment = segment[:"2017-07-30"]

        ## Check if there are any sales at all in the segment
        if segment['sales'].sum() == 0:
            zero_sales_segments.append(segment.assign(store=store, product=product_type))
        else:
            ## Filter out individual records where sales are zero
            non_zero_segment = segment[segment['sales'] > 0]
            non_zero_sales_segments.append(non_zero_segment.assign(store=store, product=product_type))

    return (pd.concat(zero_sales_segments).reset_index(),
            pd.concat(non_zero_sales_segments).reset_index())

## Main script
if __name__ == "__main__":
    filepath = "Products_Information.csv"
    product = load_and_process_data(filepath)
    segmented_data = segment_data(product)
    zero_sales_segments, non_zero_sales_segments = divide_segments(segmented_data)

    ## Example of how to use the segmented data
    print("Zero Sales Segments:")
    print("------------------------------------------------")
    print(zero_sales_segments.head())
    print("\n")
    print("Non-Zero Sales Segments:")
    print("------------------------------------------------")
    print(non_zero_sales_segments.tail())

    ## Save processed data for zero sales and non-zero sales
    # zero_sales_segments.to_csv("zero_sales_segments.csv")
    # non_zero_sales_segments.to_csv("non_zero_sales_segments.csv")

###-------------------------------------------------------------------------------
#%%#
### Step 6-3
####################################################################
### After filtering the data into zero and non-zero sales, create a heatmap to visualize the
### average sales for each store-product combination and segment the data into different groups
### based on the heatmap for the purpose of model selection in the non-zero group.
### Plot the distribution of average sales and sales range to define thresholds for each group.

## Load the data
non_product = pd.read_csv("non_zero_sales_segments.csv")

## Ensure 'product_type' is of categorical data type
non_product['product_type'] = non_product['product_type'].astype('category')

## Initialize a dictionary to hold the data for each product-store combination
segmented_data = {}

## Grouping the data by store and product, and storing each group in the dictionary
for (store, product_type), group in non_product.groupby(['store_nbr', 'product_type'], observed=True):
    segmented_data[(store, product_type)] = group[['sales', 'special_offer', 'id']]

## Splitting stores into groups
group1_stores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
group2_stores = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
group3_stores = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
group4_stores = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
group5_stores = [50, 51, 52, 53]

## Creating DataFrames for heatmaps
heatmap_data_group1 = pd.DataFrame()
heatmap_data_group2 = pd.DataFrame()
heatmap_data_group3 = pd.DataFrame()
heatmap_data_group4 = pd.DataFrame()
heatmap_data_group5 = pd.DataFrame()

## Populate the DataFrames with average sales
for (store, product_type), data in segmented_data.items():
    avg_sales = data['sales'].mean()
    if store in group1_stores:
        heatmap_data_group1.loc[store, product_type] = avg_sales
    elif store in group2_stores:
        heatmap_data_group2.loc[store, product_type] = avg_sales
    elif store in group3_stores:
        heatmap_data_group3.loc[store, product_type] = avg_sales
    elif store in group4_stores:
        heatmap_data_group4.loc[store, product_type] = avg_sales
    elif store in group5_stores:
        heatmap_data_group5.loc[store, product_type] = avg_sales

## Fill NaN values if any
heatmap_data_group1.fillna(0, inplace=True)
heatmap_data_group2.fillna(0, inplace=True)
heatmap_data_group3.fillna(0, inplace=True)
heatmap_data_group4.fillna(0, inplace=True)
heatmap_data_group5.fillna(0, inplace=True)

## Function to create heatmap
def create_heatmap(data, title, fig_size=(42, 8), cmap='Blues'):
    plt.figure(figsize=fig_size)
    sns.heatmap(data, annot=True, cmap=cmap)
    plt.title(title)
    plt.xlabel('Product Type')
    plt.ylabel('Store Number')
    plt.show()

## Creating heatmaps for each store group
group_heatmap_data = [heatmap_data_group1, heatmap_data_group2, heatmap_data_group3,
                      heatmap_data_group4, heatmap_data_group5]
for i, heatmap_data in enumerate(group_heatmap_data):
    create_heatmap(heatmap_data, f'Heatmap of Average Sales for Store Group {i+1}')

## Distribution of average sales for each store-product combination
avg_sales_data = [data['sales'].mean() for (_, data) in segmented_data.items()]
plt.figure(figsize=(12, 6))
sns.histplot(avg_sales_data, bins=30, kde=True)
plt.xlim(0, 2500)
plt.title('Distribution of Average Sales Units Across All Combinations')
plt.xlabel('Average Sales Units')
plt.ylabel('Frequency')
plt.show()

## Distribution of average sales range for each store-product combination
sales_ranges = [data['sales'].max() - data['sales'].min() for (_, data) in segmented_data.items()]
plt.figure(figsize=(12, 6))
sns.histplot(sales_ranges, bins=30, kde=True)
plt.xlim(0, 25000)
plt.title('Distribution of Sales Unit Ranges Across All Combinations')
plt.xlabel('Range of Sales Units')
plt.ylabel('Frequency')
plt.show()

###-------------------------------------------------------------------------------
#%%#
### Step 6-3 (continued)
####################################################################
### By observing the distribution of the sales unit range for each store-product combination,
### the data can be segmented into different groups based on the quantile of the sales unit range.
### Import predicting approaches and models from forecasters.py
from forecasters import SalesForecaster, ZeroSalesForecaster

###-------------------------------------------------------------------------------
#%%#
### Step 7-1
####################################################################
### Check the overall performance of using solely linear regression to predict the sales
### for each store-product combination

## Get the FIRST half of keys from the segmented data
keys = list(SalesForecaster.segmented_data.keys())
midpoint = len(keys) // 2
first_half_keys = keys[:midpoint]

## Create a DataFrame to hold the results
results_df_first_half = pd.DataFrame(columns=['Store_Number', 'Product_Type', 'Model', 'MAE',
                                               'Sales_Range', 'Average_Sales_Unit',
                                               'Predicted_Value', 'Actual_Value'])

for (store_number, product_type) in first_half_keys:
    try:
        forecaster = SalesForecaster(store_number=store_number, product_type=product_type)
        # Use the Linear Regression method
        model_used, mae, predictions, sales_range, average_sales_unit, Y_test_list = \
            forecaster.forecast_with_linear_regression()

        temp_df = pd.DataFrame([{
            'Store_Number': store_number,
            'Product_Type': product_type,
            'Model': model_used,
            'MAE': mae,
            'Sales_Range': sales_range,
            'Average_Sales_Unit': average_sales_unit,
            'Predicted_Value': predictions,
            'Actual_Value': Y_test_list
        }])
        results_df_first_half = pd.concat([results_df_first_half, temp_df], ignore_index=True)

    except Exception as e:
        print(f"Error processing store {store_number}, product {product_type}: {e}")

## Print the first few rows of the DataFrame
print(results_df_first_half.head())

###-------------------------------------------------------------------------------
#%%#
### Step 7-2
####################################################################
### Final prediction for all combinations of store and product (First Half)

## Get the FIRST half of keys from the segmented data to reduce computation time
keys = list(SalesForecaster.segmented_data.keys())
midpoint = len(keys) // 2
first_half_keys = keys[:midpoint]

## Create a DataFrame to hold the results
results_df_first_half = pd.DataFrame(columns=['Store_Number', 'Product_Type', 'Model', 'MAE',
                                               'Sales_Range', 'Average_Sales_Unit',
                                               'Predicted_Value', 'Actual_Value'])

for (store_number, product_type) in first_half_keys:
    try:
        forecaster = SalesForecaster(store_number=store_number, product_type=product_type)
        # Receive values from select_and_forecast() including model used, MAE, predictions,
        # sales range, average sales unit, and actual values for result visualization and evaluation
        model_used, mae, predictions, sales_range, average_sales_unit, Y_test_list = \
            forecaster.select_and_forecast()

        temp_df = pd.DataFrame([{
            'Store_Number': store_number,
            'Product_Type': product_type,
            'Model': model_used,
            'MAE': mae,
            'Sales_Range': sales_range,
            'Average_Sales_Unit': average_sales_unit,
            'Predicted_Value': predictions,
            'Actual_Value': Y_test_list
        }])
        results_df_first_half = pd.concat([results_df_first_half, temp_df], ignore_index=True)

    except Exception as e:
        print(f"Error processing store {store_number}, product {product_type}: {e}")

## Print the first few rows of the DataFrame
print(results_df_first_half.head())
# results_df_first_half.to_csv("results_first_half.csv")

###-------------------------------------------------------------------------------
#%%#
### Step 7-3
####################################################################
### Final prediction for all combinations of store and product (Second Half)

## Get the SECOND half of keys from the segmented data to reduce computation time
keys = list(SalesForecaster.segmented_data.keys())
midpoint = len(keys) // 2
second_half_keys = keys[midpoint:]

## Create a DataFrame to hold the results
results_df_second_half = pd.DataFrame(columns=['Store_Number', 'Product_Type', 'Model', 'MAE',
                                                'Sales_Range', 'Average_Sales_Unit',
                                                'Predicted_Value', 'Actual_Value'])

for (store_number, product_type) in second_half_keys:
    try:
        forecaster = SalesForecaster(store_number=store_number, product_type=product_type)
        # Receive values from select_and_forecast() including model used, MAE, predictions,
        # sales range, average sales unit, and actual values for result visualization and evaluation
        model_used, mae, predictions, sales_range, average_sales_unit, Y_test_list = \
            forecaster.select_and_forecast()

        temp_df = pd.DataFrame([{
            'Store_Number': store_number,
            'Product_Type': product_type,
            'Model': model_used,
            'MAE': mae,
            'Sales_Range': sales_range,
            'Average_Sales_Unit': average_sales_unit,
            'Predicted_Value': predictions,
            'Actual_Value': Y_test_list
        }])
        results_df_second_half = pd.concat([results_df_second_half, temp_df], ignore_index=True)

    except Exception as e:
        print(f"Error processing store {store_number}, product {product_type}: {e}")

## Print the first few rows of the DataFrame
print(results_df_second_half.head())
# results_df_second_half.to_csv("results_df_second_half.csv")

###-------------------------------------------------------------------------------
#%%#
### Step 7-4
####################################################################
### Result visualization and evaluation using the linear model approach
### for each store-product combination

## Load the data from CSV files
results_df_first_half = pd.read_csv("linear_results_df_first_half.csv")
results_df_second_half = pd.read_csv("linear_results_df_second_half.csv")

## Concatenate the two dataframes
linear_results_df = pd.concat([results_df_first_half, results_df_second_half], ignore_index=True)

## Display summary statistics
print("Summary Statistics:\n", linear_results_df.describe())
print('------------------------------------------------')
print('\n')
print("Summary Statistics for MAE:\n", linear_results_df['MAE'].describe())

## The number of times each model was used
model_counts = linear_results_df['Model'].value_counts()
print(model_counts)
model_counts.plot(kind='bar', figsize=(22, 6))
plt.title('Frequency of Model Selection')
plt.xlabel('Model')
plt.ylabel('Count')
plt.show()

## Check the distribution of MAE
plt.figure(figsize=(22, 6))
sns.histplot(linear_results_df['MAE'], kde=True)
plt.xlim(0, 200)
plt.title('Distribution of MAE')
plt.xlabel('MAE')
plt.ylabel('Frequency')
plt.show()

## Group the results by the model used and calculate average MAE for each model
average_mae_per_model = linear_results_df.groupby('Model')['MAE'].mean()
print("Average MAE per Model:\n", average_mae_per_model)

## Bar plot of average MAE per model
average_mae_per_model.plot(kind='bar', figsize=(22, 6))
plt.title('Average MAE per Model')
plt.xlabel('Model')
plt.ylabel('Average MAE')
plt.show()

palette1 = "rocket_r"
## Boxplot for Sales Range
plt.figure(figsize=(22, 6))
sns.boxplot(data=linear_results_df, x='Model', y='Sales_Range',
            fliersize=10, linewidth=3, saturation=0.95, palette=palette1)
plt.ylim(0, linear_results_df['Sales_Range'].quantile(0.90))
plt.title('Boxplot of Sales Range for Different Models')
plt.xlabel('Model')
plt.ylabel('Sales Range')
plt.xticks(rotation=45)
plt.show()

palette2 = "vlag"
## Boxplot for Average Sales Unit
plt.figure(figsize=(22, 6))
sns.boxplot(data=linear_results_df, x='Model', y='Average_Sales_Unit',
            fliersize=10, linewidth=3, saturation=0.95, palette=palette2)
plt.ylim(0, linear_results_df['Average_Sales_Unit'].quantile(0.90))
plt.title('Boxplot of Average Sales Unit for Different Models')
plt.xlabel('Model')
plt.ylabel('Average Sales Unit')
plt.xticks(rotation=45)
plt.show()

## Result evaluation by calculating the relative MAE
epsilon = 1e-10  # Adding a small constant to avoid division by zero

## Calculate the relative MAE
linear_results_df['Relative_MAE'] = linear_results_df['MAE'] / \
    (linear_results_df['Average_Sales_Unit'] + epsilon)

## Display the updated DataFrame with the new column
print(linear_results_df.head())

## Calculate the average relative MAE for each model
average_relative_mae_per_model = linear_results_df.groupby('Model')['Relative_MAE'].mean()

## Creating the bar plot
average_relative_mae_per_model.plot(kind='bar', figsize=(22, 6))
plt.title('Average Relative MAE per Model')
plt.xlabel('Model')
plt.ylabel('Relative MAE')
plt.show()

overall_average_relative_mae = linear_results_df['Relative_MAE'].mean()

## Creating the bar plot
plt.figure(figsize=(6, 4))
plt.bar(['Overall Average Relative MAE'], [overall_average_relative_mae])
plt.title('Overall Average Relative MAE of Linear')
plt.ylabel('Relative MAE')
plt.ylim(0, overall_average_relative_mae + (overall_average_relative_mae * 0.1))
plt.show()

print(overall_average_relative_mae)

###-------------------------------------------------------------------------------
#%%#
### Step 7-5
####################################################################
### Result visualization and evaluation using the best model selection approach
### for each store-product combination

## Load the data from CSV files
results_df_first_half = pd.read_csv("results_first_half.csv")
results_df_second_half = pd.read_csv("results_df_second_half.csv")

## Concatenate the two dataframes
results_df = pd.concat([results_df_first_half, results_df_second_half], ignore_index=True)

## Display summary statistics
print("Summary Statistics:\n", results_df.describe())
print('------------------------------------------------')
print('\n')
print("Summary Statistics for MAE:\n", results_df['MAE'].describe())

## The number of times each model was used
model_counts = results_df['Model'].value_counts()
print(model_counts)
model_counts.plot(kind='bar', figsize=(22, 6))
plt.title('Frequency of Model Selection')
plt.xlabel('Model')
plt.ylabel('Count')
plt.show()

## Check the distribution of MAE
plt.figure(figsize=(22, 6))
sns.histplot(results_df['MAE'], kde=True)
plt.xlim(0, 200)
plt.title('Distribution of MAE')
plt.xlabel('MAE')
plt.ylabel('Frequency')
plt.show()

## Group the results by the model used and calculate average MAE for each model
average_mae_per_model = results_df.groupby('Model')['MAE'].mean()
print("Average MAE per Model:\n", average_mae_per_model)

## Bar plot of average MAE per model
average_mae_per_model.plot(kind='bar', figsize=(22, 6))
plt.title('Average MAE per Model')
plt.xlabel('Model')
plt.ylabel('Average MAE')
plt.show()

palette1 = "rocket_r"
## Boxplot for Sales Range
plt.figure(figsize=(22, 6))
sns.boxplot(data=results_df, x='Model', y='Sales_Range',
            fliersize=10, linewidth=3, saturation=0.95, palette=palette1)
plt.ylim(0, results_df['Sales_Range'].quantile(0.90))
plt.title('Boxplot of Sales Range for Different Models')
plt.xlabel('Model')
plt.ylabel('Sales Range')
plt.xticks(rotation=45)
plt.show()

palette2 = "vlag"
## Boxplot for Average Sales Unit
plt.figure(figsize=(22, 6))
sns.boxplot(data=results_df, x='Model', y='Average_Sales_Unit',
            fliersize=10, linewidth=3, saturation=0.95, palette=palette2)
plt.ylim(0, results_df['Average_Sales_Unit'].quantile(0.90))
plt.title('Boxplot of Average Sales Unit for Different Models')
plt.xlabel('Model')
plt.ylabel('Average Sales Unit')
plt.xticks(rotation=45)
plt.show()

## Result evaluation by calculating the relative MAE
epsilon = 1e-10  # Adding a small constant to avoid division by zero

## Calculate the relative MAE
results_df['Relative_MAE'] = results_df['MAE'] / (results_df['Average_Sales_Unit'] + epsilon)

## Display the updated DataFrame with the new column
print(results_df.head())

## Calculate the average relative MAE for each model
average_relative_mae_per_model = results_df.groupby('Model')['Relative_MAE'].mean()

## Creating the bar plot
average_relative_mae_per_model.plot(kind='bar', figsize=(22, 6))
plt.title('Average Relative MAE per Model')
plt.xlabel('Model')
plt.ylabel('Relative MAE')
plt.show()

overall_average_relative_mae = results_df['Relative_MAE'].mean()

## Creating the bar plot
plt.figure(figsize=(6, 4))
plt.bar(['Overall Average Relative MAE'], [overall_average_relative_mae])
plt.title('Overall Average Relative MAE Across All Models')
plt.ylabel('Relative MAE')
plt.ylim(0, overall_average_relative_mae + (overall_average_relative_mae * 0.1))
plt.show()

print(overall_average_relative_mae)

###-------------------------------------------------------------------------------
#%%#
### Step 8
######################################################################
### Predict the sales unit for zero sales combinations if needed
