import pandas as pd
import numpy as np
import requests
import io

print(" Downloading FEMA disaster dataset")

fema_url = "https://www.fema.gov/api/open/v2/DisasterDeclarationsSummaries"
response = requests.get(fema_url, params={"$top": 10000})
fema_data = response.json()["DisasterDeclarationsSummaries"]

fema_df = pd.DataFrame(fema_data)[["disasterNumber","state","incidentType","fyDeclared"]]

fema_df = fema_df.rename(columns={
    "disasterNumber": "DisasterNumber", 
    "state": "State",
    "incidentType": "Peril",
    "fyDeclared": "Year"   
})

fema_df.dropna(inplace=True)

fema_df = fema_df[fema_df["Peril"].isin([
    "Hurricane", "Flood", "Tornado", "Wildfire", "Earthquake"])]

fema_df["Year"] = fema_df["Year"].astype(int)
print(f"Fema disaster dataset shape: {fema_df.shape}")


fema_df.head()

print(" Downloading NOAA storm dataset")
# ...existing code...
noaa_url = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/StormEvents_details-ftp_v1.0_d2025_c20250818.csv.gz"
noaa_response = requests.get(noaa_url)
noaa_df = pd.read_csv(io.BytesIO(noaa_response.content), compression="gzip", low_memory=False)
noaa_df = noaa_df[["STATE", "EVENT_TYPE", "DAMAGE_PROPERTY", "BEGIN_LAT", "BEGIN_LON"]].dropna()
noaa_df = noaa_df.rename(columns={
    "STATE": "State",
    "EVENT_TYPE": "Peril"})
noaa_df = noaa_df[noaa_df["Peril"].isin([
    "Hurricane", "Flood", "Tornado", "Wildfire", "Earthquake"])]
print(f"NOAA storm dataset shape: {noaa_df.shape}")

def parse_damage(val):
    try:
        val = str(val).replace("$","")
        if 'K' in val:
            return float(val.replace('K','')) * 1000
        elif 'M' in val:
            return float(val.replace('M','')) * 1000000
        elif 'B' in val:
            return float(val.replace('B','')) * 1000000000
        else:
            return float(val)
    except:
        return np.nan

noaa_df["Damage_USD"] = noaa_df["DAMAGE_PROPERTY"].apply(parse_damage)
noaa_df.drop(columns=["DAMAGE_PROPERTY"], inplace=True)
noaa_df.dropna(inplace=True)


print("Creating exposure dataset")

n = 10000
np.random.seed(42)

states = noaa_df["State"].value_counts().head(20).index.tolist()
perils = ["Hurricane", "Flood", "Tornado", "Wildfire", "Earthquake"]
exposure_data = pd.DataFrame({
    "PropertyID": [f"P{i:05d}" for i in range(1, n+1)],
    "State": np.random.choice(states, n),
    "Peril": np.random.choice(perils, n),
    "InsuredValue_USD": np.random.uniform(100000, 5000000, n),
    "ConstructionType": np.random.choice(["Wood", "Masonry", "Steel", "Concrete"], n),
    "OccupancyType": np.random.choice(["Residential", "Commercial", "Industrial"], n),
    "YearBuilt": np.random.randint(1950, 2024, n)
})
print(f"Exposure dataset shape: {exposure_data.shape}")

print("Merging datasets to create catastrophe dataset")

hazard_summary = noaa_df.groupby(["State", "Peril"])["Damage_USD"].sum().reset_index().rename(columns={"Damage_USD": "TotalDamage_USD"})

merged_df = pd.merge(exposure_data , hazard_summary, on=["State", "Peril"], how="left")

merged_df["TotalDamage_USD"].fillna(merged_df["TotalDamage_USD"].median(), inplace=True)

merged_df["HazardScore"] = np.log1p(merged_df["TotalDamage_USD"]) / np.log1p(merged_df["InsuredValue_USD"]).max()

vulnerability = np.where(merged_df["ConstructionType"] == "Wood", 0.7,
               np.where(merged_df["ConstructionType"] == "Masonry", 0.5,
               np.where(merged_df["ConstructionType"] == "Steel", 0.3, 0.2)))

sensitivity = np.where(merged_df["OccupancyType"] == "Residential", 0.6,
              np.where(merged_df["OccupancyType"] == "Commercial", 0.4, 0.3))

merged_df["Loss_Ratio"] = merged_df["HazardScore"] * vulnerability * sensitivity * np.random.uniform(0.5,1.5, len(merged_df))

merged_df["Loss_Ratio"] = merged_df["Loss_Ratio"].clip(0,1)

merged_df["Loss_Amount_USD"] = merged_df["InsuredValue_USD"] * merged_df["Loss_Ratio"]

final_df = merged_df[[
    "PropertyID", "State", "Peril", "InsuredValue_USD", 
    "ConstructionType", "OccupancyType", "YearBuilt", 
    "TotalDamage_USD", "HazardScore", "Loss_Ratio", "Loss_Amount_USD"
]]

print(f"Final catastrophe dataset shape: {final_df.shape}")

print(final_df.head())

output_path = "catastrophe_dataset.csv"
final_df.to_csv(output_path, index=False)
print(f"Catastrophe dataset saved to {output_path}")
print(f"Rows: {len(final_df)}, Columns: {len(final_df.columns)}")