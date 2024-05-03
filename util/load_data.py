import pandas as pd


def load_column(source, column_name):
    """
    Returns a file as a pandas dataframe
    """
    values = []

    with open(source, 'r') as file:
        for line in file:
            values.append(float(line.strip()))

    return pd.DataFrame({column_name: values})
