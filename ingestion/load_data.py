from pathlib import Path

import pandas as pd


DATA_FILE_PATH = Path("data/competitor_analysis_dataset.csv")


def load_competitor_data(
    file_path: Path = DATA_FILE_PATH,
) -> pd.DataFrame:
    """
    Load competitor data from a CSV file.

    Args:
        file_path: Path to the competitor CSV dataset.

    Returns:
        Pandas DataFrame containing competitor data.

    Raises:
        FileNotFoundError:
            If the CSV file does not exist.

        ValueError:
            If the CSV file is empty.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Competitor dataset not found: {file_path}"
        )

    dataframe = pd.read_csv(file_path)

    if dataframe.empty:
        raise ValueError(
            "Competitor dataset is empty."
        )

    return dataframe




if __name__ == "__main__":
    df = load_competitor_data()

    print(df.head())
    print()
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")