import os
import pandas as pd

root = r"D:\ICE interview prep\ICE IRM Modeling"
os.makedirs(f"{root}/data", exist_ok=True)

instruments = pd.DataFrame({
    "contract_code": ["TTF","NBP","BRENT","EURIBOR","SONIA"],
    "asset_class": ["Energy","Energy","Energy","Financials","Financials"],
    "contract_multiplier": [720, 300, 1000, 2500, 2500],
    "bid_ask_bps": [5.0, 6.0, 3.0, 0.8, 0.8],
    "participation_rate": [0.15, 0.12, 0.20, 0.25, 0.25],
})
inst_path = f"{root}/data/instruments.csv"
instruments.to_csv(inst_path, index=False)

positions = pd.DataFrame({
    "contract_code": ["TTF","NBP","BRENT","EURIBOR","SONIA"],
    "position_contracts": [40, -25, 15, -200, 180],
})
pos_path = f"{root}/data/positions.csv"
positions.to_csv(pos_path, index=False)
