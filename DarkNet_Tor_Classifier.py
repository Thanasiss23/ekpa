
# DarkNet_Tor_Classifier.py
# Γραμμική έκδοση - αντικαταστήστε το raw_url με το dataset URL αν χρειάζεται.

raw_url = "https://raw.githubusercontent.com/kdemertzis/EKPA/main/Data/DarkNet.csv"

import pandas as pd
df = pd.read_csv(raw_url)
print(df.shape)
print(df.columns.tolist())
