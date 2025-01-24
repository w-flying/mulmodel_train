import  pandas as pd
import numpy as np
data=pd.read_excel('../all_data.xlsx')
data.describe().to_excel('all_data_describe.xlsx')