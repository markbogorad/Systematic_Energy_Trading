import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self, sheet_name, header_row=4):
        """Loads data from an Excel sheet and cleans it."""
        df = pd.read_excel(self.file_path, sheet_name=sheet_name, header=header_row)

        # Rename first column to 'date' if necessary
        if "Dates" in df.columns:
            df.rename(columns={"Dates": "date"}, inplace=True)

        # Convert date column to datetime format
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Set 'date' as the index
        df.set_index("date", inplace=True)

        # Drop any empty rows
        df.dropna(how="all", inplace=True)

        return df
