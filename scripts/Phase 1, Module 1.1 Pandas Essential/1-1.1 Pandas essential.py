import pandas as pd

# Series
ages = pd.Series([5, 10, 15, 20], index=['Ram', 'Hari', 'Basu', 'Laxman'])
height = pd.Series([12, 14, 15, 16])

# DataFrame
data = {
    "Name": ['Ram', 'Shyam', 'Hari', 'Gopal', 'Krishna'],
    "Age": [25, 30, 35, 40, 28],
    "City": ["Kathmandu", "Pokhara", "Butwal", "Biratnagar", "Palpa"]
}
df = pd.DataFrame(data)
print(df)

# Indexing with .loc (inclusive end index)
print(f"[0:2, 'Name']: {df.loc[0:2, 'Name']}") # 0:2, 2 inclusive

# Indexing with .iloc (end index exclusive)
print(df.iloc[0]) # first row
print(df.iloc[[0, 3]]) # first and fourth row
print(df.iloc[0:3, 0:2]) # first three rows and first two columns

# chained assignment and SettingWithCopyWarning
df.loc[df['City'] == 'Butwal', 'Age'] = 50
df.loc[df['Name'] == 'Ram', 'City'] = 'Lalitpur'

df['Status'] = 'Junior'
df.loc[df['Age'] >= 30, 'Status'] = "Senior"

# MultiIndex Dataframe
df_multi = df.set_index(["City", "Name"])
print(df_multi.loc["Butwal"]) # print all rows with city == Butwal
print(df_multi.loc[("Palpa", "Krishna")]) # print all rows with city == Palpa and Name == Krishna
print(df_multi.loc[["Kathmandu", "Pokhara"]]) # print all rows for Kathmandu and Palpa at once
print(df_multi.loc["Kathmandu":"Pokhara"]) # select all rows from city == Kathmandu to city == Pokhara

df_multi.reset_index() # reset back form MultiIndex DataFrame to Normal DataFrame

