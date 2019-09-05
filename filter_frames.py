import pandas as pd
import argparse

# argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--inputFile", required=True,
	help="choose which file to filter")
ap.add_argument("-c", "--class", type=int, default=1,
	help="choose the class_id to filter for")
args = vars(ap.parse_args())

input_file = args['inputFile']
csv_name = input_file.split('/')[-1]
class_id_input = args['class']

# read in unfiltered csv as dataframe
unfiltered_df = pd.read_csv(input_file)

# function to filter
def filter_frames(data, class_id):
    # we include reset_index to ensure the indexes from original df are not used
    # drop=True param ensures index column is not included in result
    result = data.loc[data['class_id'] == class_id].loc[data.groupby(['frame'])['conf'].idxmax()].reset_index(drop=True)
    return result

filtered_df = filter_frames(unfiltered_df, class_id_input).drop(['clip', 'class_id', 'conf'], axis=1).dropna(how='all')

# print(str(type(filtered_df)))
# save to new csv
filtered_df.to_csv('./frame_data/' + csv_name, index=False)