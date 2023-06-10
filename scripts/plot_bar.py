import pandas as pd
import matplotlib.pyplot as plt
import sys

# Get the CSV file path from command line argument
csv_file = sys.argv[1]

# Read the CSV file
data = pd.read_csv(csv_file)

# Extract the column you want to plot as a list of strings
y1_label = 'f1_per_class'
y2_label = 'rmse_per_class'
y_label = y1_label
column_values_str = data[y_label].iloc[-1]

# Convert the list of strings to a list of integers
y = [float(x) for x in column_values_str.strip('[]').split(',')]
x = range(len(y))

# Create a subplot with 1 row and 1 column
fig, ax = plt.subplots(1, 1)

# Plot the column as a column graph
ax.bar(x, y)

# Display the corresponding value of each bar just above it
for i, j in zip(x, y):
    ax.text(i, j, str(j), ha='center', va='bottom')

# Customize the subplot if needed (e.g., axis labels, title, etc.)
ax.set_xticks(x)
ax.set_.ylim(0, 1)
ax.set_xlabel('Classes', labelpad=10)
ax.set_ylabel(y_label, labelpad=10)

# Display the graph
plt.show()
