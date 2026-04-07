# Author: Bansi Kamani

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


# Function to process each segment and filter out initial zero sales until the first non-zero sale
def process_segment(group):
    first_non_zero_index = group['sales'].ne(0).idxmax()
    return group.loc[first_non_zero_index:]


class ZeroSalesForecaster:
    # Load the data
    product = pd.read_csv("Products_Information.csv")

    # Convert the 'date' column to datetime format and set it as the index
    product['date'] = pd.to_datetime(product['date'])
    product.set_index('date', inplace=True)

    # Ensure 'product_type' is of categorical type
    product['product_type'] = product['product_type'].astype('category')

    # Create a dictionary for storing the store-product data sets
    segmented_data = {}

    def __init__(self, store_number, product_type, train_end_date='2016-07-31',
                 validation_end_date='2017-07-31'):
        self.store_number = store_number
        self.product_type = product_type
        self.train_end_date = train_end_date
        self.validation_end_date = validation_end_date
        self.model = None

    def predict_zero_sales(self, lags=28):
        # Get the specific segment data
        specific_segment = self.segmented_data[(self.store_number, self.product_type)]

        # The number of predictions is determined by the length of the test set
        num_predictions = len(specific_segment[self.validation_end_date:])

        # Create a zero-filled NumPy array for the predictions
        y_predict = np.zeros(num_predictions)

        # Actual sales for the test period
        Y_test = specific_segment[self.validation_end_date:]['sales']

        # Calculate MAE for zero predictions
        mae = mean_absolute_error(Y_test, y_predict)

        # Printing performance metrics
        print("Zero Sales Prediction MAE:", mae)
        daily_mae_scores = [mean_absolute_error([actual], [0]) for actual in Y_test]
        print("Day-by-Day MAE Scores:", daily_mae_scores)
        print('------------------------------------------')
        print('\n')

        return y_predict, Y_test, mae

    def select_and_forecast(self):
        # Get the specific segment data
        specific_segment = self.segmented_data[(self.store_number, self.product_type)]

        # Check if the segment has only zero sales
        if specific_segment['sales'].sum() == 0:
            return self.predict_zero_sales()
        else:
            raise ValueError("This segment has non-zero sales and is not suitable for zero sales prediction.")


class SalesForecaster:
    ## Load the data
    product = pd.read_csv("Products_Information.csv")

    ## Convert the 'date' column to datetime format and set it as the index
    product['date'] = pd.to_datetime(product['date'])
    product.set_index('date', inplace=True)

    ## Ensure 'product_type' is of categorical type
    product['product_type'] = product['product_type'].astype('category')

    ## Create a dictionary for storing the store-product data sets
    segmented_data = {}

    ## Function to process each segment if the data starts with zero sales units
    def process_segment(group):
        # Remove initial zero sales
        first_non_zero_index = group['sales'].ne(0).idxmax()
        return group.loc[first_non_zero_index:]

    # Grouping the data by store and product
    for (store, product_type), group in product.groupby(['store_nbr', 'product_type'], observed=True):
        processed_group = process_segment(group)
        segmented_data[(store, product_type)] = processed_group[['sales', 'special_offer', 'id', 'store_nbr']]

    def __init__(self, store_number, product_type, train_end_date='2016-07-31',
                 validation_end_date='2017-07-31'):
        self.store_number = store_number
        self.product_type = product_type
        self.train_end_date = train_end_date
        self.validation_end_date = validation_end_date
        self.model = None

    def forecast_with_linear_regression(self):
        # Get the specific segment data
        specific_segment = self.segmented_data[(self.store_number, self.product_type)]

        # Calculate the sales range and average sales unit
        sales_range = specific_segment['sales'].max() - specific_segment['sales'].min()
        average_sales_unit = specific_segment['sales'].mean()

        Y_test = None

        # Apply the Linear Regression model
        y_predict, Y_test, mae = self.linear_offer_date_predict()

        # Convert Y_test to a list for consistency
        Y_test_list = Y_test.tolist() if Y_test is not None else []

        return "Linear Regression", mae, y_predict, sales_range, average_sales_unit, Y_test_list

    def select_and_forecast(self):
        # Get the specific segment data
        specific_segment = self.segmented_data[(self.store_number, self.product_type)]

        # Calculate the sales range and average sales unit
        sales_range = specific_segment['sales'].max() - specific_segment['sales'].min()
        average_sales_unit = specific_segment['sales'].mean()

        # Initialize variables
        model_mae = {}
        Y_test = None

        # Function to update model_mae dictionary
        def update_model_mae(model_name, prediction_function):
            nonlocal Y_test
            y_predict, Y_test_local, mae = prediction_function()
            if y_predict is not None and Y_test_local is not None:
                model_mae[model_name] = (mae, y_predict)
                nonlocal Y_test
                Y_test = Y_test_local  # Update the Y_test in the outer scope

        # Run each model and calculate MAE
        if specific_segment['sales'].sum() == 0:
            update_model_mae('Zero Sales Prediction', self.zero_sales_predict)
        else:
            update_model_mae('Linear Regression', self.linear_offer_date_predict)
            update_model_mae('Random Forest', self.randomforest_offer_date_predict)
            update_model_mae('LightGBM', self.lightgbm_offer_date_predict)
            update_model_mae('XGBoost', self.xgboost_offer_date_predict)
            update_model_mae('MLP Regression', self.mlp_regression_offer_date_predict)

        # Select the model with the lowest MAE
        best_model, best_mae_y_predict = min(model_mae.items(), key=lambda x: x[1][0])
        best_mae, y_predict = best_mae_y_predict
        Y_test_list = Y_test.tolist() if Y_test is not None else []

        return best_model, best_mae, y_predict, sales_range, average_sales_unit, Y_test_list

    def create_sales_lag_features(self, lags=12):
        # Create lag features for the sales column using previous sales units to predict future sales
        specific_segment = self.segmented_data[(self.store_number, self.product_type)].copy()
        for lag in range(1, lags + 1):
            specific_segment[f'lag_{lag}'] = specific_segment['sales'].shift(lag)

        # Drop rows where any of the lag features is NaN
        specific_segment.dropna(subset=[f'lag_{i}' for i in range(1, lags + 1)], inplace=True)
        self.segmented_data[(self.store_number, self.product_type)] = specific_segment

    def create_offer_lag_features(self, lags=12):
        # Create lag features for the special_offer column using previous offer values to predict future sales
        specific_segment = self.segmented_data[(self.store_number, self.product_type)].copy()
        for lag in range(1, lags + 1):
            specific_segment[f'offer_lag_{lag}'] = specific_segment['special_offer'].shift(lag)

        # Drop rows where any of the lag features is NaN
        specific_segment.dropna(subset=[f'offer_lag_{i}' for i in range(1, lags + 1)], inplace=True)
        self.segmented_data[(self.store_number, self.product_type)] = specific_segment

    def linear_regression_predict(self, lags=21):
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

        # Create lag features
        self.create_sales_lag_features(lags)
        specific_segment = self.segmented_data[(self.store_number, self.product_type)]

        # Drop rows with NaN values to align X and Y
        specific_segment.dropna(subset=[f'lag_{i}' for i in range(1, lags + 1)], inplace=True)

        # Prepare the features and target variable
        lag_columns = [f'lag_{i}' for i in range(1, lags + 1)]
        X = specific_segment[lag_columns]
        Y = specific_segment['sales']

        # Split the data into training and testing sets
        X_train = X[:self.validation_end_date].values
        Y_train = Y[:self.validation_end_date]
        X_test = X[self.validation_end_date:].values
        Y_test = Y[self.validation_end_date:]

        # Initialize and train the Linear Regression model
        self.model = LinearRegression(n_jobs=-1)
        self.model.fit(X_train, Y_train)

        # Make predictions
        y_predict = self.model.predict(X_test)

        # Plotting the forecast alongside the actual test data
        dates = pd.to_datetime(Y_test.index)
        plt.figure(figsize=(22, 6))
        plt.plot(dates, y_predict, color='blue', label='Predicted Sales')
        plt.plot(dates, Y_test, color='red', label='Actual Sales')
        plt.title(f'Linear Regression Sales Forecast vs Actuals for Store {self.store_number} - Product {self.product_type}')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()

        # Evaluating the model's performance
        print("Linear Regression MSE:", mean_squared_error(Y_test, y_predict))
        print("Linear Regression R2 Score:", r2_score(Y_test, y_predict))

        # Day-by-day evaluation using MAE
        daily_mae_scores = [mean_absolute_error([actual], [predicted]) for actual, predicted in zip(Y_test, y_predict)]
        print("Day-by-Day MAE Scores:", daily_mae_scores)

    def optimize_linear_regression(self):
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import GridSearchCV, PredefinedSplit

        # Best parameters initialization
        best_mse = float('inf')
        best_lag = 0
        best_params = None

        lag_values = [14, 21, 28, 35, 42, 49, 10, 20, 30, 60, 90]

        for lag in lag_values:
            self.create_sales_lag_features(lags=lag)
            specific_segment = self.segmented_data[(self.store_number, self.product_type)].dropna()

            # Splitting the data
            train_data = specific_segment[:self.train_end_date]
            validation_data = specific_segment[self.train_end_date:]
            feature_cols = [col for col in train_data.columns if col != 'sales']
            X_train = train_data[feature_cols]
            y_train = train_data['sales']
            X_validation = validation_data[feature_cols]
            y_validation = validation_data['sales']

            # Combine train and validation sets
            X_combined = pd.concat([X_train, X_validation])
            y_combined = pd.concat([y_train, y_validation])

            # Define the split index
            split_index = [-1] * len(X_train) + [0] * len(X_validation)
            pds = PredefinedSplit(test_fold=split_index)

            # Parameter grid for Linear Regression
            param_grid = {'fit_intercept': [True, False]}

            # Grid Search with predefined split
            lr = LinearRegression(n_jobs=-1)
            grid_search = GridSearchCV(
                estimator=lr,
                param_grid=param_grid,
                cv=pds,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search.fit(X_combined, y_combined)

            # Evaluate the best model
            best_model_mse = -grid_search.best_score_
            if best_model_mse < best_mse:
                best_mse = best_model_mse
                best_lag = lag
                best_params = grid_search.best_params_

        print(f"Best MSE: {best_mse}")
        print(f"Best Lag: {best_lag}")
        print(f"Best Parameters: {best_params}")

    def linear_offer_date_plotting_predict(self, lags=28):
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

        # Create lag features for both sales and special_offer
        self.create_sales_lag_features(lags)
        self.create_offer_lag_features(lags)
        specific_segment = self.segmented_data[(self.store_number, self.product_type)]

        # Prepare the features and target variable
        sales_lag_columns = [f'lag_{i}' for i in range(1, lags + 1)]
        offer_lag_columns = [f'offer_lag_{i}' for i in range(1, lags + 1)]
        all_features = sales_lag_columns + offer_lag_columns

        # Fill NaN values with zeros in the lag features
        specific_segment[all_features] = specific_segment[all_features].fillna(0)

        X_train = specific_segment[:self.validation_end_date][all_features].values
        Y_train = specific_segment[:self.validation_end_date]['sales']
        X_test = specific_segment[self.validation_end_date:][all_features].values
        Y_test = specific_segment[self.validation_end_date:]['sales']

        # Initialize and train the Linear Regression model
        self.model = LinearRegression(n_jobs=-1)
        self.model.fit(X_train, Y_train)

        # Make predictions
        y_predict = self.model.predict(X_test)

        # Plotting the forecast alongside the actual test data
        dates = pd.to_datetime(Y_test.index)
        plt.figure(figsize=(22, 6))
        plt.plot(dates, y_predict, color='green', label='Predicted Sales')
        plt.plot(dates, Y_test, color='red', label='Actual Sales')
        plt.title(f'Linear Regression Sales Forecast vs Actuals for Store {self.store_number} - Product {self.product_type}')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()

        mae = mean_absolute_error(Y_test, y_predict)
        print("Linear Regression daily MAE:", mae)
        print("Linear Regression daily MSE:", mean_squared_error(Y_test, y_predict))
        print("Linear Regression R2 Score:", r2_score(Y_test, y_predict))

        daily_mae_scores = [mean_absolute_error([actual], [predicted]) for actual, predicted in zip(Y_test, y_predict)]
        print("Day-by-Day Linear Scores:", daily_mae_scores)

        return y_predict, Y_test, mae

    def optimize_offer_date_linear_regression(self, n_splits=5):
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import GridSearchCV

        # Best parameters initialization
        best_mse = float('inf')
        best_lag = 0
        best_params = None

        lag_values = [14, 21, 28, 35, 42, 49, 10, 20, 30, 60, 90]

        for lag in lag_values:
            self.create_sales_lag_features(lags=lag)
            self.create_offer_lag_features(lags=lag)
            specific_segment = self.segmented_data[(self.store_number, self.product_type)]
            specific_segment = specific_segment[:self.validation_end_date].dropna()

            # Prepare the data
            feature_cols = [col for col in specific_segment.columns if col != 'sales']
            X = specific_segment[feature_cols]
            y = specific_segment['sales']

            # Parameter grid for Linear Regression
            param_grid = {}

            # Time Series Cross-Validator
            tscv = TimeSeriesSplit(n_splits=n_splits)

            # Grid Search with time series cross-validation
            lr_model = LinearRegression(n_jobs=-1)
            grid_search = GridSearchCV(
                estimator=lr_model,
                param_grid=param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search.fit(X, y)

            # Evaluate the best model
            best_model_mse = -grid_search.best_score_
            if best_model_mse < best_mse:
                best_mse = best_model_mse
                best_lag = lag
                best_params = grid_search.best_params_

        print(f"Best MSE: {best_mse}")
        print(f"Best Lag: {best_lag}")
        print(f"Best Parameters: {best_params}")

    # Fine-tuned by store1-EGGS (sales scale: 60-140)
    def mlp_regression_offer_date_plotting_predict(self, lags=21):
        from sklearn.neural_network import MLPRegressor
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

        # Create lag features for both sales and special_offer
        self.create_sales_lag_features(lags)
        self.create_offer_lag_features(lags)
        specific_segment = self.segmented_data[(self.store_number, self.product_type)]

        # Prepare the features and target variable
        sales_lag_columns = [f'lag_{i}' for i in range(1, lags + 1)]
        offer_lag_columns = [f'offer_lag_{i}' for i in range(1, lags + 1)]
        all_features = sales_lag_columns + offer_lag_columns

        # Fill NaN values with zeros in the lag features
        specific_segment[all_features] = specific_segment[all_features].fillna(0)

        X_train = specific_segment[:self.validation_end_date][all_features].values
        Y_train = specific_segment[:self.validation_end_date]['sales']
        X_test = specific_segment[self.validation_end_date:][all_features].values
        Y_test = specific_segment[self.validation_end_date:]['sales']

        # Initialize and train the MLP Regressor model
        self.model = MLPRegressor(
            hidden_layer_sizes=(200, 200, 200, 200, 100),
            max_iter=500,
            learning_rate='adaptive',
            random_state=42
        )
        self.model.fit(X_train, Y_train)

        # Make predictions
        y_predict = self.model.predict(X_test)

        # Plotting the forecast alongside the actual test data
        dates = pd.to_datetime(Y_test.index)
        plt.figure(figsize=(22, 6))
        plt.plot(dates, y_predict, color='green', label='Predicted Sales')
        plt.plot(dates, Y_test, color='red', label='Actual Sales')
        plt.title(f'MLP Regression Sales Forecast vs Actuals for Store {self.store_number} - Product {self.product_type}')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()

        mae = mean_absolute_error(Y_test, y_predict)
        print("MLP Regression daily MAE:", mae)
        print("MLP Regression daily MSE:", mean_squared_error(Y_test, y_predict))
        print("MLP Regression R2 Score:", r2_score(Y_test, y_predict))

        daily_mae_scores = [mean_absolute_error([actual], [predicted]) for actual, predicted in zip(Y_test, y_predict)]
        print("Day-by-Day MAE Scores:", daily_mae_scores)

        return y_predict, Y_test, mae

    def optimize_offer_date_mlp_regression(self):
        from sklearn.neural_network import MLPRegressor
        from sklearn.model_selection import GridSearchCV, PredefinedSplit

        # Best parameters initialization
        best_mse = float('inf')
        best_lag = 0
        best_params = None

        lag_values = [14, 21, 28, 35, 42, 49, 10, 20, 30, 60, 90]

        for lag in lag_values:
            self.create_sales_lag_features(lags=lag)
            specific_segment = self.segmented_data[(self.store_number, self.product_type)].dropna()

            # Splitting the data
            train_data = specific_segment[:self.train_end_date]
            validation_data = specific_segment[self.train_end_date:]
            feature_cols = [col for col in train_data.columns if col != 'sales']
            X_train = train_data[feature_cols]
            y_train = train_data['sales']
            X_validation = validation_data[feature_cols]
            y_validation = validation_data['sales']

            # Combine train and validation sets
            X_combined = pd.concat([X_train, X_validation])
            y_combined = pd.concat([y_train, y_validation])

            # Define the split index
            split_index = [-1] * len(X_train) + [0] * len(X_validation)
            pds = PredefinedSplit(test_fold=split_index)

            # Parameter grid for MLP Regression
            param_grid = {
                'hidden_layer_sizes': [(200, 200, 100), (200, 200, 200, 100), (200, 200, 200, 200, 100)],
                'learning_rate': ['adaptive', 'constant', 'invscaling'],
            }

            # Grid Search with predefined split
            mlp = MLPRegressor(max_iter=500, early_stopping=True, random_state=42)
            grid_search = GridSearchCV(
                estimator=mlp,
                param_grid=param_grid,
                cv=pds,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search.fit(X_combined, y_combined)

            # Evaluate the best model
            best_model_mse = -grid_search.best_score_
            if best_model_mse < best_mse:
                best_mse = best_model_mse
                best_lag = lag
                best_params = grid_search.best_params_

        print(f"Best MSE: {best_mse}")
        print(f"Best Lag: {best_lag}")
        print(f"Best Parameters: {best_params}")

    # Fine-tuned by store4-DAIRY (sales scale: 500-1100)
    def randomforest_date_predict(self, lags=14):
        # Create lag features for both sales and special_offer
        self.create_sales_lag_features(lags)
        self.create_offer_lag_features(lags)
        specific_segment = self.segmented_data[(self.store_number, self.product_type)]
        specific_segment.dropna(inplace=True)

        # Prepare the features and target variable
        sales_lag_columns = [f'lag_{i}' for i in range(1, lags + 1)]
        offer_lag_columns = [f'offer_lag_{i}' for i in range(1, lags + 1)]
        all_features = sales_lag_columns + offer_lag_columns

        X_train = specific_segment[:self.validation_end_date][all_features].values
        Y_train = specific_segment[:self.validation_end_date]['sales']
        X_test = specific_segment[self.validation_end_date:][all_features].values
        Y_test = specific_segment[self.validation_end_date:]['sales']

        # Train the Random Forest model
        self.model = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=34, n_jobs=-1)
        self.model.fit(X_train, Y_train)
        rf_predictions = self.model.predict(X_test)

        # Plotting the forecast alongside the actual test data
        dates = pd.to_datetime(Y_test.index)
        plt.figure(figsize=(22, 6))
        plt.plot(dates, rf_predictions, color='blue', label='RF Predicted Sales')
        plt.plot(dates, Y_test, color='red', label='Actual Sales')
        plt.title(f'Random Forest Sales Forecast vs Actuals for Store {self.store_number} - Product {self.product_type}')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()

        mae = mean_absolute_error(Y_test, rf_predictions)
        print("Random Forest MAE:", mae)
        print("Random Forest MSE:", mean_squared_error(Y_test, rf_predictions))
        print("Random Forest R2 Score:", r2_score(Y_test, rf_predictions))

        daily_mae_scores = [mean_absolute_error([actual], [predicted]) for actual, predicted in zip(Y_test, rf_predictions)]
        print("Day-by-Day RF Scores:", daily_mae_scores)

        return rf_predictions, Y_test, mae

    def randomforest_offer_date_plotting_predict(self, lags=14):
        # Create lag features for both sales and special_offer
        self.create_sales_lag_features(lags)
        self.create_offer_lag_features(lags)
        specific_segment = self.segmented_data[(self.store_number, self.product_type)]
        specific_segment.dropna(inplace=True)

        # Prepare the features and target variable
        sales_lag_columns = [f'lag_{i}' for i in range(1, lags + 1)]
        offer_lag_columns = [f'offer_lag_{i}' for i in range(1, lags + 1)]
        all_features = sales_lag_columns + offer_lag_columns

        X_train = specific_segment[:self.validation_end_date][all_features].values
        Y_train = specific_segment[:self.validation_end_date]['sales']
        X_test = specific_segment[self.validation_end_date:][all_features].values
        Y_test = specific_segment[self.validation_end_date:]['sales']

        # Train the Random Forest model
        self.model = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=34, n_jobs=-1)
        self.model.fit(X_train, Y_train)
        rf_predictions = self.model.predict(X_test)

        # Plotting the forecast alongside the actual test data
        dates = pd.to_datetime(Y_test.index)
        plt.figure(figsize=(22, 6))
        plt.plot(dates, rf_predictions, color='blue', label='RF Predicted Sales')
        plt.plot(dates, Y_test, color='red', label='Actual Sales')
        plt.title(f'Random Forest Sales Forecast vs Actuals for Store {self.store_number} - Product {self.product_type}')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()

        mae = mean_absolute_error(Y_test, rf_predictions)
        print("Random Forest MAE:", mae)
        print("Random Forest MSE:", mean_squared_error(Y_test, rf_predictions))
        print("Random Forest R2 Score:", r2_score(Y_test, rf_predictions))

        daily_mae_scores = [mean_absolute_error([actual], [predicted]) for actual, predicted in zip(Y_test, rf_predictions)]
        print("Day-by-Day RF Scores:", daily_mae_scores)

        return rf_predictions, Y_test, mae

    def optimize_date_randomforest(self):
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import GridSearchCV, PredefinedSplit

        # Best parameters initialization
        best_mse = float('inf')
        best_lag = 0
        best_params = None

        lag_values = [14, 21, 28, 35, 42, 49]

        for lag in lag_values:
            self.create_sales_lag_features(lags=lag)
            specific_segment = self.segmented_data[(self.store_number, self.product_type)].dropna()

            # Splitting the data
            train_data = specific_segment[:self.train_end_date]
            validation_data = specific_segment[self.train_end_date:]
            feature_cols = [col for col in train_data.columns if col != 'sales']
            X_train = train_data[feature_cols]
            y_train = train_data['sales']
            X_validation = validation_data[feature_cols]
            y_validation = validation_data['sales']

            # Combine train and validation sets
            X_combined = pd.concat([X_train, X_validation])
            y_combined = pd.concat([y_train, y_validation])

            # Define the split index
            split_index = [-1] * len(X_train) + [0] * len(X_validation)
            pds = PredefinedSplit(test_fold=split_index)

            # Parameter grid for Random Forest
            param_grid = {
                'n_estimators': [100, 200, 250, 300, 350, 400],
                'max_depth': [20, 30, 40, 50],
                'n_jobs': [-1]
            }

            # Grid Search with cross-validation
            self.model = RandomForestRegressor(random_state=34)
            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                cv=pds,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search.fit(X_combined, y_combined)

            # Evaluate the best model
            best_model_mse = -grid_search.best_score_
            if best_model_mse < best_mse:
                best_mse = best_model_mse
                best_lag = lag
                best_params = grid_search.best_params_

        print(f"Best MSE: {best_mse}")
        print(f"Best Lag: {best_lag}")
        print(f"Best Parameters: {best_params}")

    def lightgbm_date_predict(self, lags=14):
        from sklearn.metrics import r2_score, mean_absolute_error

        # Create lag features using create_sales_lag_features(), adding previous sales units as features
        self.create_sales_lag_features(lags)
        specific_segment = self.segmented_data[(self.store_number, self.product_type)]

        # Prepare the features and target variable
        lag_columns = [f'lag_{i}' for i in range(1, lags + 1)]
        X_train = specific_segment[:self.validation_end_date][lag_columns].values
        Y_train = specific_segment[:self.validation_end_date]['sales']
        X_test = specific_segment[self.validation_end_date:][lag_columns].values
        Y_test = specific_segment[self.validation_end_date:]['sales']

        # Initialize and train the LightGBM model
        self.model = lgb.LGBMRegressor(
            n_estimators=100, max_depth=20, random_state=42, learning_rate=0.05, n_jobs=-1
        )
        self.model.fit(X_train, Y_train)

        # Make predictions
        y_predict = self.model.predict(X_test)

        # Plotting the forecast alongside the actual test data
        dates = pd.to_datetime(Y_test.index)
        plt.figure(figsize=(22, 6))
        plt.plot(dates, y_predict, color='blue', label='Predicted Sales')
        plt.plot(dates, Y_test, color='red', label='Actual Sales')
        plt.title(f'LightGBM-PureDate Sales Forecast vs Actuals for Store {self.store_number} - Product {self.product_type}')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()

        print("LightGBM r2_score:", r2_score(Y_test, y_predict))
        daily_mae_scores = [mean_absolute_error([actual], [predicted]) for actual, predicted in zip(Y_test, y_predict)]
        print("Day-by-Day MAE Scores:", daily_mae_scores)

    def lightgbm_offer_date_plotting_predict(self, lags=14):
        from sklearn.metrics import r2_score, mean_absolute_error

        # Create lag features for both sales and special_offer to improve model performance
        self.create_sales_lag_features(lags)
        self.create_offer_lag_features(lags)
        specific_segment = self.segmented_data[(self.store_number, self.product_type)]

        # Prepare the features and target variable
        sales_lag_columns = [f'lag_{i}' for i in range(1, lags + 1)]
        offer_lag_columns = [f'offer_lag_{i}' for i in range(1, lags + 1)]
        all_features = sales_lag_columns + offer_lag_columns

        X_train = specific_segment[:self.validation_end_date][all_features].values
        Y_train = specific_segment[:self.validation_end_date]['sales']
        X_test = specific_segment[self.validation_end_date:][all_features].values
        Y_test = specific_segment[self.validation_end_date:]['sales']

        # Initialize and train the LightGBM model
        self.model = lgb.LGBMRegressor(
            n_estimators=400, max_depth=30, random_state=42, learning_rate=0.01, n_jobs=-1
        )
        self.model.fit(X_train, Y_train)

        # Make predictions
        y_predict = self.model.predict(X_test)

        # Plotting the forecast alongside the actual test data
        dates = pd.to_datetime(Y_test.index)
        plt.figure(figsize=(22, 6))
        plt.plot(dates, y_predict, color='green', label='Predicted Sales')
        plt.plot(dates, Y_test, color='red', label='Actual Sales')
        plt.title(f'LightGBM Sales Forecast vs Actuals for Store {self.store_number} - Product {self.product_type}')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()

        mae = mean_absolute_error(Y_test, y_predict)
        print("LightGBM MAE:", mae)
        print("LightGBM MSE:", mean_squared_error(Y_test, y_predict))
        print("LightGBM R2 Score:", r2_score(Y_test, y_predict))

        daily_mae_scores = [mean_absolute_error([actual], [predicted]) for actual, predicted in zip(Y_test, y_predict)]
        print("Day-by-Day LightGBM Scores:", daily_mae_scores)

        return y_predict, Y_test, mae

    def lightgbm_date_predict_rolling(self, lags=14, forecast_horizon=16):
        from sklearn.metrics import r2_score, mean_absolute_error

        # Create lag features
        self.create_sales_lag_features(lags)
        specific_segment = self.segmented_data[(self.store_number, self.product_type)].dropna()

        # Prepare the features and target variable
        lag_columns = [f'lag_{i}' for i in range(1, lags + 1)]
        X = specific_segment[lag_columns]
        Y = specific_segment['sales']

        # Split data into train and forecast sets
        X_train = X[:-forecast_horizon]
        Y_train = Y[:-forecast_horizon]
        X_forecast = X[-forecast_horizon:]

        # Initialize and train the LightGBM model
        self.model = lgb.LGBMRegressor(
            n_estimators=100, max_depth=20, random_state=42, learning_rate=0.05, n_jobs=-1
        )
        self.model.fit(X_train, Y_train)

        # Rolling forecasting
        predictions = []
        for i in range(forecast_horizon):
            next_day_prediction = self.model.predict(X_forecast.iloc[[i]].values)
            predictions.append(next_day_prediction[0])
            if i < forecast_horizon - 1:
                X_forecast.iloc[i + 1, -lags:-1] = X_forecast.iloc[i, -(lags - 1):]

        predictions = np.array(predictions)
        Y_test = Y[-forecast_horizon:]

        # Plotting the forecast alongside the actual test data
        dates = pd.to_datetime(Y_test.index)
        plt.figure(figsize=(22, 6))
        plt.plot(dates, predictions, color='blue', label='Predicted Sales')
        plt.plot(dates, Y_test, color='red', label='Actual Sales')
        plt.title(f'LightGBM Rolling Forecast vs Actuals for Store {self.store_number} - Product {self.product_type}')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()

        print("LightGBM Rolling r2_score:", r2_score(Y_test, predictions))
        daily_mae_scores = [mean_absolute_error([actual], [predicted]) for actual, predicted in zip(Y_test, predictions)]
        print("Day-by-Day MAE Scores:", daily_mae_scores)

    def lightgbm_EXfeature_predict(self):
        from sklearn.metrics import r2_score, mean_absolute_error

        specific_segment = self.segmented_data[(self.store_number, self.product_type)]

        # Prepare the features and target variable using external features
        X_train = specific_segment[:self.validation_end_date][['special_offer', 'id', 'store_nbr']].values
        Y_train = specific_segment[:self.validation_end_date]['sales']
        X_test = specific_segment[self.validation_end_date:][['special_offer', 'id', 'store_nbr']].values
        Y_test = specific_segment[self.validation_end_date:]['sales']

        # Initialize and train the LightGBM model
        self.model = lgb.LGBMRegressor(
            n_estimators=100, max_depth=10, random_state=42, learning_rate=0.1
        )
        self.model.fit(X_train, Y_train)

        # Make predictions
        y_predict = self.model.predict(X_test)

        # Plotting the forecast alongside the actual test data
        dates = pd.to_datetime(Y_test.index)
        plt.figure(figsize=(22, 6))
        plt.plot(dates, y_predict, color='blue', label='Predicted Sales')
        plt.plot(dates, Y_test, color='red', label='Actual Sales')
        plt.title(f'LightGBM-EXfeatures Sales Forecast vs Actuals for Store {self.store_number} - Product {self.product_type}')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()

        print("LightGBM r2_score:", r2_score(Y_test, y_predict))
        daily_mae_scores = [mean_absolute_error([actual], [predicted]) for actual, predicted in zip(Y_test, y_predict)]
        print("Day-by-Day MAE Scores:", daily_mae_scores)

    def optimize_date_lightgbm(self):
        from sklearn.model_selection import GridSearchCV, PredefinedSplit

        # Best parameters initialization
        best_mse = float('inf')
        best_lag = 0
        best_params = None

        lag_values = [14, 21, 28, 35, 42, 49, 30, 60, 90]

        for lag in lag_values:
            self.create_sales_lag_features(lags=lag)
            specific_segment = self.segmented_data[(self.store_number, self.product_type)].dropna()

            # Splitting the data
            train_data = specific_segment[:self.train_end_date]
            validation_data = specific_segment[self.train_end_date:]
            feature_cols = [col for col in train_data.columns if col != 'sales']
            X_train = train_data[feature_cols]
            y_train = train_data['sales']
            X_validation = validation_data[feature_cols]
            y_validation = validation_data['sales']

            # Combine train and validation sets
            X_combined = pd.concat([X_train, X_validation])
            y_combined = pd.concat([y_train, y_validation])

            # Define the split index
            split_index = [-1] * len(X_train) + [0] * len(X_validation)
            pds = PredefinedSplit(test_fold=split_index)

            # Parameter grid for LightGBM
            param_grid = {
                'n_estimators': [100, 200, 250, 300, 350, 400],
                'max_depth': [20, 30, 40, 50, 60],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_jobs': [-1]
            }

            # Grid Search with predefined split
            self.model = lgb.LGBMRegressor(random_state=42, force_col_wise=True)
            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                cv=pds,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search.fit(X_combined, y_combined)

            # Evaluate the best model
            best_model_mse = -grid_search.best_score_
            if best_model_mse < best_mse:
                best_mse = best_model_mse
                best_lag = lag
                best_params = grid_search.best_params_

        print(f"Best MSE: {best_mse}")
        print(f"Best Lag: {best_lag}")
        print(f"Best Parameters: {best_params}")

    def optimize_offer_date_lightgbm(self, n_splits=5):
        from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

        # Best parameters initialization
        best_mse = float('inf')
        best_lag = 0
        best_params = None

        lag_values = [14, 21, 28, 35, 42, 49, 10, 20, 30, 60, 90]

        for lag in lag_values:
            # Create lag features for sales and special offers
            self.create_sales_lag_features(lags=lag)
            self.create_offer_lag_features(lags=lag)
            specific_segment = self.segmented_data[(self.store_number, self.product_type)]
            specific_segment = specific_segment[:self.validation_end_date].dropna()

            # Prepare the data
            feature_cols = [col for col in specific_segment.columns if col != 'sales']
            X = specific_segment[feature_cols]
            y = specific_segment['sales']

            # Parameter grid for LightGBM
            param_grid = {
                'n_estimators': [100, 200, 250, 300, 350, 400],
                'max_depth': [20, 30, 40, 50, 60],
                'learning_rate': [0.01, 0.05, 0.1]
            }

            # Time Series Cross-Validator
            tscv = TimeSeriesSplit(n_splits=n_splits)

            # Grid Search with time series cross-validation
            lgb_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1, force_col_wise=True)
            grid_search = GridSearchCV(
                estimator=lgb_model,
                param_grid=param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search.fit(X, y)

            # Evaluate the best model
            best_model_mse = -grid_search.best_score_
            if best_model_mse < best_mse:
                best_mse = best_model_mse
                best_lag = lag
                best_params = grid_search.best_params_

        print(f"Best MSE: {best_mse}")
        print(f"Best Lag: {best_lag}")
        print(f"Best Parameters: {best_params}")

    # Fine-tuned by store3-POULTRY (sales scale: 800-1600)
    def xgboost_date_predict(self, lags=28):
        from sklearn.metrics import r2_score, mean_absolute_error

        # Create lag features
        self.create_sales_lag_features(lags)
        specific_segment = self.segmented_data[(self.store_number, self.product_type)]

        # Prepare the features and target variable
        lag_columns = [f'lag_{i}' for i in range(1, lags + 1)]
        X_train = specific_segment[:self.validation_end_date][lag_columns].values
        Y_train = specific_segment[:self.validation_end_date]['sales']
        X_test = specific_segment[self.validation_end_date:][lag_columns].values
        Y_test = specific_segment[self.validation_end_date:]['sales']

        # Initialize and train the XGBoost model
        self.model = xgb.XGBRegressor(
            n_estimators=200, max_depth=50, random_state=42, learning_rate=0.01, n_jobs=-1
        )
        self.model.fit(X_train, Y_train)

        # Make predictions
        y_predict = self.model.predict(X_test)

        # Plotting the forecast alongside the actual test data
        dates = pd.to_datetime(Y_test.index)
        plt.figure(figsize=(22, 6))
        plt.plot(dates, y_predict, color='blue', label='Predicted Sales')
        plt.plot(dates, Y_test, color='red', label='Actual Sales')
        plt.title(f'XGBoost-PureDate Sales Forecast vs Actuals for Store {self.store_number} - Product {self.product_type}')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()

        print("XGBoost r2_score:", r2_score(Y_test, y_predict))
        daily_mae_scores = [mean_absolute_error([actual], [predicted]) for actual, predicted in zip(Y_test, y_predict)]
        print("Day-by-Day MAE Scores:", daily_mae_scores)

    def xgboost_offer_date_plotting_predict(self, lags=28):
        from sklearn.metrics import r2_score, mean_absolute_error

        # Create lag features for both sales and special_offer
        self.create_sales_lag_features(lags)
        self.create_offer_lag_features(lags)
        specific_segment = self.segmented_data[(self.store_number, self.product_type)]

        # Prepare the features and target variable
        sales_lag_columns = [f'lag_{i}' for i in range(1, lags + 1)]
        offer_lag_columns = [f'offer_lag_{i}' for i in range(1, lags + 1)]
        all_features = sales_lag_columns + offer_lag_columns

        X_train = specific_segment[:self.validation_end_date][all_features].values
        Y_train = specific_segment[:self.validation_end_date]['sales']
        X_test = specific_segment[self.validation_end_date:][all_features].values
        Y_test = specific_segment[self.validation_end_date:]['sales']

        # Initialize and train the XGBoost model
        self.model = xgb.XGBRegressor(
            n_estimators=200, max_depth=50, random_state=42, learning_rate=0.01, n_jobs=-1
        )
        self.model.fit(X_train, Y_train)

        # Make predictions
        y_predict = self.model.predict(X_test)

        # Plotting the forecast alongside the actual test data
        dates = pd.to_datetime(Y_test.index)
        plt.figure(figsize=(22, 6))
        plt.plot(dates, y_predict, color='blue', label='Predicted Sales')
        plt.plot(dates, Y_test, color='red', label='Actual Sales')
        plt.title(f'XGBoost Date & Offer Sales Forecast vs Actuals for Store {self.store_number} - Product {self.product_type}')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()

        mae = mean_absolute_error(Y_test, y_predict)
        print("XGBoost Date & Offer MAE:", mae)
        print("XGBoost Date & Offer MSE:", mean_squared_error(Y_test, y_predict))
        print("XGBoost Date & Offer R2 Score:", r2_score(Y_test, y_predict))

        daily_mae_scores = [mean_absolute_error([actual], [predicted]) for actual, predicted in zip(Y_test, y_predict)]
        print("Day-by-Day MAE Scores:", daily_mae_scores)

        return y_predict, Y_test, mae

    def xgboost_date_predict_rolling(self, lags=28, forecast_horizon=16):
        from sklearn.metrics import r2_score, mean_absolute_error

        # Create lag features
        self.create_sales_lag_features(lags)
        specific_segment = self.segmented_data[(self.store_number, self.product_type)].dropna()

        # Prepare the features and target variable
        lag_columns = [f'lag_{i}' for i in range(1, lags + 1)]
        X = specific_segment[lag_columns]
        Y = specific_segment['sales']

        # Split data into train and forecast sets
        X_train = X[:-forecast_horizon]
        Y_train = Y[:-forecast_horizon]
        X_forecast = X[-forecast_horizon:]

        # Initialize and train the XGBoost model
        self.model = xgb.XGBRegressor(
            n_estimators=200, max_depth=50, random_state=42, learning_rate=0.01, n_jobs=-1
        )
        self.model.fit(X_train, Y_train)

        # Rolling forecasting
        predictions = []
        for i in range(forecast_horizon):
            next_day_prediction = self.model.predict(X_forecast.iloc[[i]].values)
            predictions.append(next_day_prediction[0])
            if i < forecast_horizon - 1:
                X_forecast.iloc[i + 1, -lags:-1] = X_forecast.iloc[i, -(lags - 1):]

        predictions = np.array(predictions)
        Y_test = Y[-forecast_horizon:]

        # Plotting the forecast alongside the actual test data
        dates = pd.to_datetime(Y_test.index)
        plt.figure(figsize=(22, 6))
        plt.plot(dates, predictions, color='blue', label='Predicted Sales')
        plt.plot(dates, Y_test, color='red', label='Actual Sales')
        plt.title(f'XGBoost Rolling Forecast vs Actuals for Store {self.store_number} - Product {self.product_type}')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.legend()
        plt.show()

        print("XGBoost Rolling r2_score:", r2_score(Y_test, predictions))
        daily_mae_scores = [mean_absolute_error([actual], [predicted]) for actual, predicted in zip(Y_test, predictions)]
        print("Day-by-Day MAE Scores:", daily_mae_scores)

    def optimize_date_xgboost(self):
        from sklearn.model_selection import GridSearchCV, PredefinedSplit

        # Best parameters initialization
        best_mse = float('inf')
        best_lag = 0
        best_params = None

        lag_values = [14, 21, 28, 35, 42, 49, 30, 60, 90]

        for lag in lag_values:
            self.create_sales_lag_features(lags=lag)
            specific_segment = self.segmented_data[(self.store_number, self.product_type)].dropna()

            # Splitting the data
            train_data = specific_segment[:self.train_end_date]
            validation_data = specific_segment[self.train_end_date:]
            feature_cols = [col for col in train_data.columns if col != 'sales']
            X_train = train_data[feature_cols]
            y_train = train_data['sales']
            X_validation = validation_data[feature_cols]
            y_validation = validation_data['sales']

            # Combine train and validation sets
            X_combined = pd.concat([X_train, X_validation])
            y_combined = pd.concat([y_train, y_validation])

            # Define the split index
            split_index = [-1] * len(X_train) + [0] * len(X_validation)
            pds = PredefinedSplit(test_fold=split_index)

            # Parameter grid for XGBoost
            param_grid = {
                'n_estimators': [100, 200, 250, 300, 350, 400],
                'max_depth': [30, 40, 50, 60],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_jobs': [-1]
            }

            # Grid Search with cross-validation
            self.model = xgb.XGBRegressor(random_state=42)
            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                cv=pds,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search.fit(X_combined, y_combined)

            # Evaluate the best model
            best_model_mse = -grid_search.best_score_
            if best_model_mse < best_mse:
                best_mse = best_model_mse
                best_lag = lag
                best_params = grid_search.best_params_

        print(f"Best MSE: {best_mse}")
        print(f"Best Lag: {best_lag}")
        print(f"Best Parameters: {best_params}")

    def optimize_offer_date_xgboost(self, n_splits=5):
        from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

        # Best parameters initialization
        best_mse = float('inf')
        best_lag = 0
        best_params = None

        lag_values = [14, 21, 28, 35, 42, 49, 30, 60, 90]

        for lag in lag_values:
            # Create lag features for sales and special offers
            self.create_sales_lag_features(lags=lag)
            self.create_offer_lag_features(lags=lag)
            specific_segment = self.segmented_data[(self.store_number, self.product_type)]
            specific_segment = specific_segment[:self.validation_end_date].dropna()

            # Prepare the data
            feature_cols = [col for col in specific_segment.columns if col != 'sales']
            X = specific_segment[feature_cols]
            y = specific_segment['sales']

            # Parameter grid for XGBoost
            param_grid = {
                'n_estimators': [100, 200, 250, 300, 350, 400],
                'max_depth': [30, 40, 50, 60],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_jobs': [-1]
            }

            # Time Series Cross-Validator
            tscv = TimeSeriesSplit(n_splits=n_splits)

            # Grid Search with time series cross-validation
            xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(
                estimator=xgb_model,
                param_grid=param_grid,
                cv=tscv,
                scoring='neg_mean_squared_error'
            )
            grid_search.fit(X, y)

            # Evaluate the best model
            best_model_mse = -grid_search.best_score_
            if best_model_mse < best_mse:
                best_mse = best_model_mse
                best_lag = lag
                best_params = grid_search.best_params_

        print(f"Best MSE: {best_mse}")
        print(f"Best Lag: {best_lag}")
        print(f"Best Parameters: {best_params}")

    # Fine-tuned by store18-DAIRY (sales scale: 400-900)
    def linear_offer_date_predict(self, lags=28):
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

        # Create lag features for both sales and special_offer
        self.create_sales_lag_features(lags)
        self.create_offer_lag_features(lags)
        specific_segment = self.segmented_data[(self.store_number, self.product_type)]

        # Prepare the features and target variable
        sales_lag_columns = [f'lag_{i}' for i in range(1, lags + 1)]
        offer_lag_columns = [f'offer_lag_{i}' for i in range(1, lags + 1)]
        all_features = sales_lag_columns + offer_lag_columns

        # Fill NaN values with zeros in the lag features
        specific_segment[all_features] = specific_segment[all_features].fillna(0)

        X_train = specific_segment[:self.validation_end_date][all_features].values
        Y_train = specific_segment[:self.validation_end_date]['sales']
        X_test = specific_segment[self.validation_end_date:][all_features].values
        Y_test = specific_segment[self.validation_end_date:]['sales']

        # Initialize and train the Linear Regression model
        self.model = LinearRegression(n_jobs=-1)
        self.model.fit(X_train, Y_train)

        # Make predictions
        y_predict = self.model.predict(X_test)
        mae = mean_absolute_error(Y_test, y_predict)

        print("Linear Regression daily MAE:", mae)
        print("Linear Regression daily MSE:", mean_squared_error(Y_test, y_predict))
        print("Linear Regression R2 Score:", r2_score(Y_test, y_predict))

        daily_mae_scores = [mean_absolute_error([actual], [predicted]) for actual, predicted in zip(Y_test, y_predict)]
        print("Day-by-Day Linear Scores:", daily_mae_scores)
        print('------------------------------------------')
        print('\n')

        return y_predict, Y_test, mae

    def randomforest_offer_date_predict(self, lags=14):
        # Create lag features for both sales and special_offer
        self.create_sales_lag_features(lags)
        self.create_offer_lag_features(lags)
        specific_segment = self.segmented_data[(self.store_number, self.product_type)]
        specific_segment.dropna(inplace=True)

        # Prepare the features and target variable
        sales_lag_columns = [f'lag_{i}' for i in range(1, lags + 1)]
        offer_lag_columns = [f'offer_lag_{i}' for i in range(1, lags + 1)]
        all_features = sales_lag_columns + offer_lag_columns

        X_train = specific_segment[:self.validation_end_date][all_features].values
        Y_train = specific_segment[:self.validation_end_date]['sales']
        X_test = specific_segment[self.validation_end_date:][all_features].values
        Y_test = specific_segment[self.validation_end_date:]['sales']

        # Train the Random Forest model
        self.model = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=34, n_jobs=-1)
        self.model.fit(X_train, Y_train)
        rf_predictions = self.model.predict(X_test)

        mae = mean_absolute_error(Y_test, rf_predictions)
        print("Random Forest MAE:", mae)
        print("Random Forest MSE:", mean_squared_error(Y_test, rf_predictions))
        print("Random Forest R2 Score:", r2_score(Y_test, rf_predictions))

        daily_mae_scores = [mean_absolute_error([actual], [predicted]) for actual, predicted in zip(Y_test, rf_predictions)]
        print("Day-by-Day RF Scores:", daily_mae_scores)
        print('------------------------------------------')
        print('\n')

        return rf_predictions, Y_test, mae

    def mlp_regression_offer_date_predict(self, lags=21):
        from sklearn.neural_network import MLPRegressor
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

        # Create lag features for both sales and special_offer
        self.create_sales_lag_features(lags)
        self.create_offer_lag_features(lags)
        specific_segment = self.segmented_data[(self.store_number, self.product_type)]

        # Prepare the features and target variable
        sales_lag_columns = [f'lag_{i}' for i in range(1, lags + 1)]
        offer_lag_columns = [f'offer_lag_{i}' for i in range(1, lags + 1)]
        all_features = sales_lag_columns + offer_lag_columns

        # Fill NaN values with zeros in the lag features
        specific_segment[all_features] = specific_segment[all_features].fillna(0)

        X_train = specific_segment[:self.validation_end_date][all_features].values
        Y_train = specific_segment[:self.validation_end_date]['sales']
        X_test = specific_segment[self.validation_end_date:][all_features].values
        Y_test = specific_segment[self.validation_end_date:]['sales']

        # Initialize and train the MLP Regressor model
        self.model = MLPRegressor(
            hidden_layer_sizes=(200, 200, 200, 200, 100),
            max_iter=500,
            learning_rate='adaptive',
            random_state=42
        )
        self.model.fit(X_train, Y_train)

        # Make predictions
        y_predict = self.model.predict(X_test)
        mae = mean_absolute_error(Y_test, y_predict)

        print("MLP Regression daily MAE:", mae)
        print("MLP Regression daily MSE:", mean_squared_error(Y_test, y_predict))
        print("MLP Regression R2 Score:", r2_score(Y_test, y_predict))

        daily_mae_scores = [mean_absolute_error([actual], [predicted]) for actual, predicted in zip(Y_test, y_predict)]
        print("Day-by-Day MAE Scores:", daily_mae_scores)
        print('------------------------------------------')
        print('\n')

        return y_predict, Y_test, mae

    def lightgbm_offer_date_predict(self, lags=14):
        from sklearn.metrics import r2_score, mean_absolute_error

        # Create lag features for both sales and special_offer to improve model performance
        self.create_sales_lag_features(lags)
        self.create_offer_lag_features(lags)
        specific_segment = self.segmented_data[(self.store_number, self.product_type)]

        # Prepare the features and target variable
        sales_lag_columns = [f'lag_{i}' for i in range(1, lags + 1)]
        offer_lag_columns = [f'offer_lag_{i}' for i in range(1, lags + 1)]
        all_features = sales_lag_columns + offer_lag_columns

        X_train = specific_segment[:self.validation_end_date][all_features].values
        Y_train = specific_segment[:self.validation_end_date]['sales']
        X_test = specific_segment[self.validation_end_date:][all_features].values
        Y_test = specific_segment[self.validation_end_date:]['sales']

        # Initialize and train the LightGBM model
        self.model = lgb.LGBMRegressor(
            n_estimators=400, max_depth=30, random_state=42, learning_rate=0.01, n_jobs=-1
        )
        self.model.fit(X_train, Y_train)

        # Make predictions
        y_predict = self.model.predict(X_test)
        mae = mean_absolute_error(Y_test, y_predict)

        print("LightGBM MAE:", mae)
        print("LightGBM MSE:", mean_squared_error(Y_test, y_predict))
        print("LightGBM R2 Score:", r2_score(Y_test, y_predict))

        daily_mae_scores = [mean_absolute_error([actual], [predicted]) for actual, predicted in zip(Y_test, y_predict)]
        print("Day-by-Day LightGBM Scores:", daily_mae_scores)
        print('------------------------------------------')
        print('\n')

        return y_predict, Y_test, mae

    def xgboost_offer_date_predict(self, lags=28):
        from sklearn.metrics import r2_score, mean_absolute_error

        # Create lag features for both sales and special_offer
        self.create_sales_lag_features(lags)
        self.create_offer_lag_features(lags)
        specific_segment = self.segmented_data[(self.store_number, self.product_type)]

        # Prepare the features and target variable
        sales_lag_columns = [f'lag_{i}' for i in range(1, lags + 1)]
        offer_lag_columns = [f'offer_lag_{i}' for i in range(1, lags + 1)]
        all_features = sales_lag_columns + offer_lag_columns

        X_train = specific_segment[:self.validation_end_date][all_features].values
        Y_train = specific_segment[:self.validation_end_date]['sales']
        X_test = specific_segment[self.validation_end_date:][all_features].values
        Y_test = specific_segment[self.validation_end_date:]['sales']

        # Initialize and train the XGBoost model
        self.model = xgb.XGBRegressor(
            n_estimators=200, max_depth=50, random_state=42, learning_rate=0.01, n_jobs=-1
        )
        self.model.fit(X_train, Y_train)

        # Make predictions
        y_predict = self.model.predict(X_test)
        mae = mean_absolute_error(Y_test, y_predict)

        print("XGBoost Date & Offer MAE:", mae)
        print("XGBoost Date & Offer MSE:", mean_squared_error(Y_test, y_predict))
        print("XGBoost Date & Offer R2 Score:", r2_score(Y_test, y_predict))

        daily_mae_scores = [mean_absolute_error([actual], [predicted]) for actual, predicted in zip(Y_test, y_predict)]
        print("Day-by-Day MAE Scores:", daily_mae_scores)
        print('------------------------------------------')
        print('\n')

        return y_predict, Y_test, mae

# %%
