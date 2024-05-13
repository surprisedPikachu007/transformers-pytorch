from utils import *

# download_data(data_folder="data")

prepare_data(data_folder="data",
             min_length=3,
             max_length=150,
             max_length_ratio=2.,
             retain_case=True)
