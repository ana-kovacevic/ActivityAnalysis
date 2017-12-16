import pandas as pd

# read data
data = pd.read_csv('activities_out.csv')

# get variables
df = data[['user_in_role_id', 'detection_variable_name','interval_end', 'measure_value']]

# normalise (Feature scaling)
normalized = df.groupby(['user_in_role_id','detection_variable_name']).transform(lambda x: (x - x.min()) / (x.max()-x.min()))

# add normalised vale to data
df['Normalised'] = normalized

# write data to csv
df.to_csv('check.csv')
