import pandas as pd

dataset_path = ".docs/test_FD001.txt"

def load_dataset(dataset_path:str,verbose:bool=False)->pd.DataFrame:
    df = pd.read_csv(dataset_path, sep=" ", names=[
        'unit_number','time_cycle','setting_1','setting_2','setting_3',
        't2','t24','t30','t50','p2','p15','p30','Nf','Nc','ept','Ps30',
        'phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd',
        'w31','w32'
    ])
    if verbose == True:
        print(df)
    return df
def save_dataset():
    
    return

load_dataset(dataset_path,verbose=True)
