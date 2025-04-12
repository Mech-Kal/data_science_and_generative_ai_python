# Q1. Demonstrate three different methods for creating identical 2D arrays in NumPy

import numpy as np

# Method 1: Using np.array directly
arr1 = np.array([[1, 2], [3, 4]])
print("Method 1:\n", arr1)

# Method 2: Using np.reshape on a flat list
arr2 = np.reshape([1, 2, 3, 4], (2, 2))
print("Method 2:\n", arr2)

# Method 3: Using np.full with manual assignment
arr3 = np.empty((2, 2))
arr3[0][0], arr3[0][1], arr3[1][0], arr3[1][1] = 1, 2, 3, 4
print("Method 3:\n", arr3)
# bash end

# Q2. Generate an array of 100 evenly spaced numbers between 1 and 10 and reshape into 2D

arr = np.linspace(1, 10, 100).reshape(10, 10)
print(arr)
# bash end

# Q3. Generate a 3x3 array with random floating-point numbers between 5 and 20 and round to 2 decimals

arr = np.random.uniform(5, 20, (3, 3))
arr_rounded = np.round(arr, 2)
print(arr_rounded)
# bash end

# Q4. Create a NumPy array with random integers between 1 and 10 of shape (5, 6)
#     Extract all even and all odd integers from the array

arr = np.random.randint(1, 11, (5, 6))
even = arr[arr % 2 == 0]
odd = arr[arr % 2 != 0]
print("Array:\n", arr)
print("Even numbers:", even)
print("Odd numbers:", odd)
# bash end

# Q5. Create a 3D array of shape (3, 3, 3) with integers from 1 to 10
#     a) Indices of max values along 3rd axis
#     b) Element-wise multiplication with itself

arr = np.random.randint(1, 11, (3, 3, 3))
max_indices = np.argmax(arr, axis=2)
elementwise_product = arr * arr
print("Array:\n", arr)
print("Max indices:\n", max_indices)
print("Element-wise Product:\n", elementwise_product)
# bash end

# Q6. Clean and transform the 'Phone' column in the sample dataset
import pandas as pd

df = pd.read_csv("people.csv")  # Assuming 'people.csv' is available locally
df['Phone'] = df['Phone'].astype(str).str.replace(r'\D', '', regex=True)
df['Phone'] = pd.to_numeric(df['Phone'], errors='coerce')
print(df.dtypes)
# bash end

# Q7. Perform the following on people dataset:
# a) Skip first 50 rows
# b) Read specific columns
# c) Display first 10 rows
# d) Show last 5 values of 'Salary'

df_filtered = pd.read_csv("people.csv", skiprows=50, usecols=['Last Name', 'Gender', 'Email', 'Phone', 'Salary'])
print(df_filtered.head(10))
print(df_filtered['Salary'].tail(5))
# bash end

# Q8. Filter People_Dataset where Last Name = 'Duke', Gender = 'Female', and Salary < 85000

filtered = df[(df['Last Name'].str.contains("Duke", case=False, na=False)) &
              (df['Gender'].str.contains("Female", case=False, na=False)) &
              (df['Salary'] < 85000)]
print(filtered)
# bash end

# Q9. Create a 7x5 DataFrame using a Series of 35 random integers between 1 and 6

series_data = pd.Series(np.random.randint(1, 7, 35))
df_random = series_data.values.reshape(7, 5)
df_random = pd.DataFrame(df_random, columns=[f'Col{i+1}' for i in range(5)])
print(df_random)
# bash end


# Q1. Create two Series of length 50 and join into DataFrame with renamed columns

import pandas as pd
import numpy as np

series1 = pd.Series(np.random.randint(10, 51, 50))
series2 = pd.Series(np.random.randint(100, 1001, 50))

df = pd.concat([series1, series2], axis=1)
df.columns = ['col1', 'col2']
print(df.head())
# bash end

# Q2. Perform the following on people dataset:
# a) Delete 'Email', 'Phone', 'Date of birth'
# b) Drop rows with any NaN values

people_df = pd.read_csv("people.csv")  # Assuming the file is present
people_df = people_df.drop(['Email', 'Phone', 'Date of birth'], axis=1, errors='ignore')
people_df_cleaned = people_df.dropna()
print(people_df_cleaned.head())
# bash end

# Q3. Scatter plot with horizontal & vertical lines, labels, title, and legend

import matplotlib.pyplot as plt

x = np.random.rand(100)
y = np.random.rand(100)

plt.scatter(x, y, color='red', marker='o', label='Data Points')
plt.axhline(0.5, color='blue', linestyle='--', label='y = 0.5')
plt.axvline(0.5, color='green', linestyle=':', label='x = 0.5')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Advanced Scatter Plot of Random Values')
plt.legend()
plt.grid(True)
plt.show()
# bash end

# Q4. Create time-series DataFrame and dual-axis plot for Temperature and Humidity

date_rng = pd.date_range(start='2023-01-01', periods=100, freq='D')
temp = np.random.randint(15, 35, 100)
humidity = np.random.randint(40, 90, 100)
df_weather = pd.DataFrame({'Date': date_rng, 'Temperature': temp, 'Humidity': humidity})

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(df_weather['Date'], df_weather['Temperature'], 'g-', label='Temperature')
ax2.plot(df_weather['Date'], df_weather['Humidity'], 'b-', label='Humidity')

ax1.set_xlabel('Date')
ax1.set_ylabel('Temperature', color='g')
ax2.set_ylabel('Humidity', color='b')
plt.title('Temperature and Humidity Over Time')
fig.tight_layout()
plt.show()
# bash end

# Q5. Histogram with PDF overlay from 1000 normal samples

from scipy.stats import norm

data = np.random.randn(1000)
count, bins, ignored = plt.hist(data, bins=30, density=True, alpha=0.6, color='orange', label='Histogram')

pdf = norm.pdf(bins, np.mean(data), np.std(data))
plt.plot(bins, pdf, 'k-', linewidth=2, label='Normal PDF')
plt.xlabel('Value')
plt.ylabel('Frequency/Probability')
plt.title('Histogram with PDF Overlay')
plt.legend()
plt.grid(True)
plt.show()
# bash end

# Q6. Seaborn scatter plot colored by quadrant with legend and labels

import seaborn as sns

x = np.random.uniform(-10, 10, 100)
y = np.random.uniform(-10, 10, 100)

quadrants = []
for i in range(len(x)):
    if x[i] >= 0 and y[i] >= 0:
        quadrants.append('Q1')
    elif x[i] < 0 and y[i] >= 0:
        quadrants.append('Q2')
    elif x[i] < 0 and y[i] < 0:
        quadrants.append('Q3')
    else:
        quadrants.append('Q4')

df_quadrants = pd.DataFrame({'x': x, 'y': y, 'Quadrant': quadrants})

sns.scatterplot(data=df_quadrants, x='x', y='y', hue='Quadrant', palette='Set1')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Quadrant-wise Scatter Plot')
plt.legend()
plt.grid(True)
plt.show()
# bash end


# 1. Using Bokeh, plot a line chart of a sine wave function, add grid lines, label the axes, and set the title as 'Sine Wave Function'

from bokeh.plotting import figure, show
import numpy as np

# Generate sine wave data
x = np.linspace(0, 4 * np.pi, 100)
y = np.sin(x)

# Create a figure
p = figure(title="Sine Wave Function", x_axis_label='X', y_axis_label='Y')

# Add a line renderer
p.line(x, y, legend_label="Sine Wave", line_width=2)

# Add grid lines
p.grid.grid_line_alpha = 0.3

# Show the plot
show(p)

# 2. Using Bokeh, generate a bar chart of randomly generated categorical data, color bars based on their values, 
#    add hover tooltips to display exact values, label the axes, and set the title as 'Random Categorical Bar Chart'

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool
import random

# Generate random categorical data
categories = ['A', 'B', 'C', 'D', 'E']
values = [random.randint(10, 100) for _ in categories]

# Create a data source
source = ColumnDataSource(data=dict(categories=categories, values=values))

# Create a figure
p = figure(x_range=categories, title="Random Categorical Bar Chart", x_axis_label='Category', y_axis_label='Value')

# Add bar renderer
p.vbar(x='categories', top='values', width=0.9, source=source, legend_field="categories", line_color="white", fill_color="blue")

# Add hover tooltips
hover = HoverTool()
hover.tooltips = [("Category", "@categories"), ("Value", "@values")]
p.add_tools(hover)

# Show the plot
show(p)


# 3. Using Plotly, create a basic line plot of a randomly generated dataset, label the axes, and set the title as 'Simple Line Plot'

import plotly.graph_objects as go
import numpy as np

# Generate random data
x = np.linspace(0, 10, 100)
y = np.random.random(100)

# Create a line plot
fig = go.Figure(data=go.Scatter(x=x, y=y, mode='lines'))

# Set labels and title
fig.update_layout(title='Simple Line Plot', xaxis_title='X', yaxis_title='Y')

# Show the plot
fig.show()


# 4. Using Plotly, create an interactive pie chart of randomly generated data, add labels and percentages, 
#    set the title as 'Interactive Pie Chart'

import plotly.graph_objects as go

# Random data for pie chart
labels = ['A', 'B', 'C', 'D', 'E']
values = [15, 30, 45, 5, 5]

# Create a pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hoverinfo='label+percent', textinfo='label+percent')])

# Set the title
fig.update_layout(title='Interactive Pie Chart')

# Show the plot
fig.show()


