import pandas as pd
import numpy as np

if __name__ == "__main__":
    fullData = pd.read_csv('eph_results.csv')
    mapNames = set(fullData['mapName'])

    for mapName in mapNames:
        mapSubset = fullData[fullData['mapName'] == mapName].reset_index(drop=True)
        mapSubset["Program"] = "EPH"
        mapSubset = mapSubset.rename(columns={"mapName": "Map_Name", 
                                    "agentNum": "Agent_Size",
                                    "success": "Success_Rate",
                                    "total_cost_true": "Solution_Cost",
                                    "runtime": "Runtime"})
        mapSubset[["Agent_Size","Program","Success_Rate","Solution_Cost","Runtime","Map_Name","Program"]].to_csv(f"eph_results/{mapName}.csv")

        # mapName,agentNum,success,total_cost_true,runtime
        # Agent_Size,Program,Success_Rate,Solution_Cost,Runtime,Map_Name