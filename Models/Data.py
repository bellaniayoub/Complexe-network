import pandas as pd
import numpy as np
import networkx as nx

class Data(pd.DataFrame):

    def __init__(self,csv_file):
        """
            Open the file and check if it contain the needed data
        """
        try:
            self.data = pd.read_csv(csv_file, header=None, names=['Source', 'Target'])
        except Exception as e:
            print("Cannot open the file or the file contain data that's not compatible")

        

        
    
    def extractGraph(self, df):
        G = nx.from_pandas_edgelist(self.data)
        