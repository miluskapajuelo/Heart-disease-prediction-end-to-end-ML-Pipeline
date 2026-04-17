from pathlib import Path
import pandas as pd


def load_data(file_path:str | Path) -> pd.DataFrame:
    """
    Load a dataset from a CVS file.

    Args: 
        file_path: Path to the CSV file.

    Returns: 
        A pandas Dataframe containing the loaded data.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the loaded dataframe is empty.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(f"The file is empty: {file_path}")
    
    return df


