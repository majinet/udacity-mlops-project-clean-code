"""
This is the Python Test for the churn_library.py module.

This module will be used to test
    1. import_data
    2. peform_eda
    3. encode_data
    4. perform_feature_engineering
    5. train_test_model

Author: Atauro Chow
Date: 4-Jan-2023
"""

import pandas as pd
import logging
import churn_library as cls
from os.path import exists


def test_import_data():
    """
    test data import - this example is completed for you to assist with the other test functions

    :return:
    """

    logging.info("Testing import_data")

    df = cls.import_data("./data/bank_data.csv")

    assert df.shape[0] > 0, "import data error: no row was found"
    assert df.shape[1] > 0, "import data error: no column was found"

    logging.info("Testing import_data: SUCCESS")


def test_eda():
    """
    test perform eda function

    :return:
    """

    logging.info("Testing eda")

    df = pd.read_csv("./data/bank_data.csv")
    cls.perform_eda(df)

    assert exists('./images/eda/Customer_Age.png'), "Customer_Age.png not found"
    assert exists('./images/eda/Marital_Status.png'), "Marital_Status.png not found"
    assert exists('./images/eda/Churn.png'), "Churn.png not found"

    logging.info("Testing perform_eda: SUCCESS")


def test_encoder_helper():
    """
    test encoder helper

    :return:
    """

    logging.info("Testing encoder_helper")

    # encoded column
    category_lst = [
        'Gender'
    ]

    df = pd.read_csv("./data/bank_data.csv")

    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

    result_df = cls.encoder_helper(df, category_lst, 'Churn')

    assert result_df['Gender_Churn'].shape[0] > 0, "Testing encoder_helper: The dataframe doesn't appear to have rows"

    logging.info("Testing encoder_helper: SUCCESS")


def test_perform_feature_engineering():
    """
    test perform_feature_engineering
    :return:
    """

    logging.info("Testing perform_feature_engineering")

    df = pd.read_csv("./data/bank_data.csv")
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(df, 'Churn')

    assert X_train.shape[0] > 0, "The X_train doesn't appear to have rows"
    assert X_train.shape[1] > 0, "The X_train doesn't appear to have columns"
    assert X_test.shape[0] > 0, "The X_test doesn't appear to have rows"
    assert X_test.shape[1] > 0, "The X_test doesn't appear to have columns"
    assert y_train.shape[0] > 0, "The y_train doesn't appear to have rows"
    assert y_test.shape[0] > 0, "The y_test doesn't appear to have columns"

    logging.info("Testing perform_feature_engineering: SUCCESS")


def test_train_models():
    """
    test train_models

    :return:
    """

    logging.info("Testing train_models")

    X_train_lst = [[45, 3, 39, 5, 1, 3, 12691.0, 777, 11914.0, 1.335, 1144, 42, 1.625, 0.061, 1, 1, 1, 1, 1],
                    [49, 5, 44, 6, 1, 2, 8256.0, 864, 7392.0, 1.541, 1291, 33, 3.714, 0.105, 1, 1, 1, 1, 1],
                    [51, 3, 36, 4, 1, 0, 3418.0, 0, 3418.0, 2.594, 1887, 20, 2.333, 0.0, 1, 1, 1, 1, 1],
                    [40, 4, 34, 3, 4, 1, 3313.0, 2517, 796.0, 1.405, 1171, 20, 2.333, 0.76, 1, 1, 1, 1, 1],
                    [62, 0, 49, 2, 3, 3, 1438.3, 0, 1438.3, 1.047, 692, 16, 0.6, 0.0, 1, 1, 1, 1, 1],
                    [45, 3, 39, 5, 1, 3, 12691.0, 777, 11914.0, 1.335, 1144, 42, 1.625, 0.061, 1, 1, 1, 1, 1],
                    [49, 5, 44, 6, 1, 2, 8256.0, 864, 7392.0, 1.541, 1291, 33, 3.714, 0.105, 1, 1, 1, 1, 1],
                    [51, 3, 36, 4, 1, 0, 3418.0, 0, 3418.0, 2.594, 1887, 20, 2.333, 0.0, 1, 1, 1, 1, 1],
                    [40, 4, 34, 3, 4, 1, 3313.0, 2517, 796.0, 1.405, 1171, 20, 2.333, 0.76, 1, 1, 1, 1, 1],
                    [62, 0, 49, 2, 3, 3, 1438.3, 0, 1438.3, 1.047, 692, 16, 0.6, 0.0, 1, 1, 1, 1, 1]]
    X_train = pd.DataFrame(X_train_lst,
            columns=['Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon',
                     'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1',
                     'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                     'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 'Income_Category_Churn', 'Card_Category_Churn'
                     ])

    y_train_lst = [[0],
               [0],
               [0],
               [0],
               [1],
               [0],
               [0],
               [0],
               [0],
               [1]]

    y_train = pd.DataFrame(y_train_lst, columns=['Churn'])

    X_test_lst = [[45, 3, 39, 5, 1, 3, 12691.0, 777, 11914.0, 1.335, 1144, 42, 1.625, 0.061, 1, 1, 1, 1, 1],
               [49, 5, 44, 6, 1, 2, 8256.0, 864, 7392.0, 1.541, 1291, 33, 3.714, 0.105, 1, 1, 1, 1, 1],
               [51, 3, 36, 4, 1, 0, 3418.0, 0, 3418.0, 2.594, 1887, 20, 2.333, 0.0, 1, 1, 1, 1, 1],
               [40, 4, 34, 3, 4, 1, 3313.0, 2517, 796.0, 1.405, 1171, 20, 2.333, 0.76, 1, 1, 1, 1, 1],
               [62, 0, 49, 2, 3, 3, 1438.3, 0, 1438.3, 1.047, 692, 16, 0.6, 0.0, 1, 1, 1, 1, 1],
               [45, 3, 39, 5, 1, 3, 12691.0, 777, 11914.0, 1.335, 1144, 42, 1.625, 0.061, 1, 1, 1, 1, 1],
               [49, 5, 44, 6, 1, 2, 8256.0, 864, 7392.0, 1.541, 1291, 33, 3.714, 0.105, 1, 1, 1, 1, 1],
               [51, 3, 36, 4, 1, 0, 3418.0, 0, 3418.0, 2.594, 1887, 20, 2.333, 0.0, 1, 1, 1, 1, 1],
               [40, 4, 34, 3, 4, 1, 3313.0, 2517, 796.0, 1.405, 1171, 20, 2.333, 0.76, 1, 1, 1, 1, 1],
               [62, 0, 49, 2, 3, 3, 1438.3, 0, 1438.3, 1.047, 692, 16, 0.6, 0.0, 1, 1, 1, 1, 1]]

    X_test = pd.DataFrame(X_test_lst,
                           columns=['Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count',
                                    'Months_Inactive_12_mon',
                                    'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 'Avg_Open_To_Buy',
                                    'Total_Amt_Chng_Q4_Q1',
                                    'Total_Trans_Amt', 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                                    'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                                    'Income_Category_Churn', 'Card_Category_Churn'
                                    ])

    y_test_lst = [[0],
               [0],
               [0],
               [0],
               [1],
               [0],
               [0],
               [0],
               [0],
               [1]]

    y_test = pd.DataFrame(y_test_lst, columns=['Churn'])

    cls.train_models(X_train, X_test, y_train, y_test)

    assert exists('./models/logistic_model.pkl'), "LogisticRegression models cannot be generated"
    assert exists('./models/rfc_model.pkl'), "RandomForestClassifier models cannot be generated"

    logging.info("Testing train_models: SUCCESS")
