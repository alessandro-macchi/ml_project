import pandas as pd

def load_and_combine_data(red_path: str, white_path: str) -> pd.DataFrame:
    red = pd.read_csv(red_path, sep=';')
    white = pd.read_csv(white_path, sep=';')

    red["white_type"] = 0  # red
    white["white_type"] = 1  # white

    data = pd.concat([red, white], ignore_index=True)
    return data