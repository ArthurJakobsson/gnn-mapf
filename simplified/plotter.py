import pandas as pd
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

# Load the dataset (replace with the actual path and format of your dataset)
# Assuming the dataset is in CSV format with columns: 'numAgents', 'Real-Time-LaCAM', 'CS-PIBT'
data_path = '/home/ubuntu/research/gnn-mapf/logs/random_32_32_20.csv'
data = pd.read_csv(data_path)
data = data[data['agentNum'] <= 350]

# Group data by 'numAgents' and calculate the mean success rate for each method
data = data[['agentNum', 'shieldType', 'success']]
data1 = data[data['shieldType'] == 'Real-Time-LaCAM']
data2 = data[data['shieldType'] == 'CS-PIBT']
data1 = data1.groupby('agentNum')['success'].mean()
data2 = data2.groupby('agentNum')['success'].mean()

# Plot the success rate for each method
plt.plot(data1.index, data1, marker="o", label='SSIL + Real-Time LaCAM')
plt.plot(data2.index, data2, marker='o', label='SSIL + CS-PIBT')
plt.ylim(0, 1.05)
plt.xlabel('Number of Agents')
plt.ylabel('Success Rate')
plt.legend()

# Save the plot to a file
plt.savefig('success_rate.png', bbox_inches='tight', dpi=300)