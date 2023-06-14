sns.heatmap(data = df.isna(),yticklabels=False)

df['highschool'].value_counts(dropna=False)
df['highschool'].fillna("Others",inplace=True)

df['highschool'] = df['highschool'].astype("category")
df['highschool'].value_counts()

df['URM'].value_counts()

others = {"idk? lol":"No","-":"No"}
df['URM'] = df['URM'].replace(others)
df['URM'].value_counts()

df['faaltu'].value_counts(dropna=False)

df['faaltu'].fillna("No Response",inplace=True)
df['faaltu'].astype("category")

df['edaccept'].value_counts(dropna=False)

other_college = {"N/a":"Others","N/A":"Others","nah":"Others"}

df['edaccept'] = df['edaccept'].replace(other_college)
df['edaccept'].fillna("Others",inplace=True)

df['attending'].value_counts(dropna=False)

df['attending'].fillna("Not sure yet",inplace=True)
attending = {"Idk prob'ly Reed":"Reed" }
df['attending'] = df['attending'].replace(attending)

df.columns = df.columns.str.strip()
df['Add. Info/Context'].fillna("No Response",inplace=True)
df['acceptance'].fillna("No Response",inplace=True)
sns.heatmap(data = df.isna(),yticklabels=False)
