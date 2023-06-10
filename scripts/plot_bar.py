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

# Set up the axes with gridspec
fig = plt.figure(figsize=(12, 6))
grid = plt.GridSpec(2, 2, hspace=0.4, wspace=0.4)
main_ax = fig.add_subplot(grid[0, 0])
y_hist = fig.add_subplot(grid[0, 1])
x_hist = fig.add_subplot(grid[-1, :])

# Plot the column as a column graph
main_ax.bar(x, y)
y_hist.bar(x, y)

# Display the corresponding value of each bar just above it
for i, j in zip(x, y):
    main_ax.text(i, j, str(round(j, 2)), ha='center', va='bottom')

# Customize the subplot if needed (e.g., axis labels, title, etc.)
main_ax.set_xticks(x)
main_ax.set_ylim(0, 1)
main_ax.set_xlabel('Classes')  #, labelpad=10)
main_ax.set_ylabel(y_label)  #, labelpad=10)

# Display the graph
plt.show()
