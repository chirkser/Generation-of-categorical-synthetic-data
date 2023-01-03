from helpful_functions import generate_data
from os import path
import torch
import pandas as pd
import sys

def main():
    args = sys.argv
    if len(args) != 3:
        print('Incorrect number of arguments.')
        return
      
    model_name = args[1]
    if model_name != 'CTGAN' and model_name != 'TVAE' and model_name != 'CopulaGan' and model_name != 'GaussianCopula':
      print('Unknown model name')
      return 
    path_to_data = args[2]
    num_rows = args[3]
    
    orig_data = pd.read_csv(path_to_data)
    synt_data = generate_data(orig_data, model_name, num_rows)
    synt_data.to_csv('synthetic_data.csv')
    
if __name__ == "__main__":
    main()
