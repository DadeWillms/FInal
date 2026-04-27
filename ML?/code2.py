import pandas as pd

files = ["1.xlsx","2.xlsx","3.xlsx","4.xlsx","5.xlsx","6.xlsx","7.xlsx","8.xlsx","9.xlsx"]

dfs = []

for i, file in enumerate(files):
    
    raw = pd.read_excel(file, header=None)
    
    # flatten everything to search safely
    flat = raw.astype(str)
    
    room = None
    people = None
    max_people = None
    
    for r in range(raw.shape[0]):
        for c in range(raw.shape[1]):
            val = str(raw.iloc[r, c]).strip().lower()
            
            if "wh" in val:   # room like WH286A
                room = raw.iloc[r, c]

            if "te" in val:   # room like WH286A
                room = raw.iloc[r, c]
            
            if "number of people" in val:
                people = raw.iloc[r, c+1] if c+1 < raw.shape[1] else None
            
            if "max" in val:
                max_people = raw.iloc[r, c+1] if c+1 < raw.shape[1] else None
    
    df = raw.iloc[:, :4].copy()
    df.columns = ["time", "co2", "temp", "humidity"]
    df = df.dropna()
    
    df["room"] = room
    df["people"] = people
    df["max_people"] = max_people
    df["dataset_id"] = i
    
    dfs.append(df)

final_df = pd.concat(dfs, ignore_index=True)
final_df.to_csv("clean_data.csv", index=False)