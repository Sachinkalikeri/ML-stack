import pandas as pd

annually_df = pd.read_csv('/kaggle/input/deaths-by-cancer-in-europe/tps00116.tsv', sep='\t')

pivot_data_col = annually_df.columns[0]
time_columns = annually_df.columns[1:]

annually_df['unit'] = annually_df[pivot_data_col].apply(lambda x: x.split(",")[0])
annually_df['age'] = annually_df[pivot_data_col].apply(lambda x: x.split(",")[1])
annually_df['sex'] = annually_df[pivot_data_col].apply(lambda x: x.split(",")[2])
annually_df['geography'] = annually_df[pivot_data_col].apply(lambda x: x.split(",")[3])

selected_columns = list(['unit', 'age', 'sex', 'geography']) + list(time_columns)
annually_df = annually_df[selected_columns]

annually_tr_df = annually_df.melt(id_vars=['unit', 'age', 'sex', 'geography'],
                                  var_name="date",
                                  value_name="value")
annually_tr_df['date'] = annually_tr_df['date'].apply(lambda x: int(x))
annually_tr_df['value'] = annually_tr_df['value'].apply(lambda x: str(x).replace("d", ""))
annually_tr_df['value'] = annually_tr_df['value'].apply(lambda x: str(x).replace("p", ""))
annually_tr_df['value'] = annually_tr_df['value'].apply(lambda x: str(x).replace(": ", "NAN"))
annually_tr_df['value'] = annually_tr_df['value'].apply(lambda x: float(x))

