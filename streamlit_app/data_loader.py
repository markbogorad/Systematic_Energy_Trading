import pandas as pd

def load_commodity_data(filepath):
    xl = pd.ExcelFile(filepath)
    sheets = xl.sheet_names

    commodity_data = {}
    for sheet in sheets:
        try:
            df = xl.parse(sheet, header=4)
            df = df.rename(columns={df.columns[0]: 'Date'})
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])

            df.columns = ['Date'] + [f"{sheet}{i+1} Comdty" for i in range(len(df.columns) - 1)]
            df.set_index('Date', inplace=True)

            commodity_data[sheet] = df
        except Exception as e:
            print(f"⚠️ Could not parse sheet {sheet}: {e}")

    return commodity_data