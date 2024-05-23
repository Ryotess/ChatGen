import pandas as pd


def load_xlsx(file_path: str, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(file_path, sheet_name)


def create_input_data(data: pd.DataFrame) -> list:
    levels = ["A", "B", "C", "Z"]
    return [
        data.loc[
            data["Level"] == level,
            ["UID", "Parent", "Well-formed questions", "Well-formed answers"],
        ].values
        for level in levels
    ]
