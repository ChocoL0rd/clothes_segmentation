import os
import json
import pandas as pd

def read_metric_history(path: str):
    """
    Reads all metrics{number}.xlsx files and returns dictionary {number: metric_df, ...} 
    """
    epoch2metric_df = {}
    for file_name in os.listdir(path):
        if file_name.startswith("metrics") and i.endswith(".xlsx"):
            epoch_num = int(re.search(r"\d+", i).group())
            df = pd.read_excel(os.path.join(path, file_name))
            epoch2metric_df[epoch_num] = df

    return metric_pairs 



def metrics2history(epoch2metric_df: dict, agg: dict):
    for i in sorted(epoch2metric_df.keys()):
        metric_dfs.append(
            epoch2metric_df[i].\
            groupby("dir").\
            aggregate(agg).\
            transpose()
        )
    
    return pd.concat(metric_dfs)



