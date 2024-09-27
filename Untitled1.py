#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Load the penguins dataset using seaborn
penguins = sns.load_dataset('penguins')

# Create the figure and axes
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Iterate over each species
for i, (species, data) in enumerate(penguins.groupby('species')):
    # Create the KDE plot
    sns.kdeplot(data['flipper_length_mm'], ax=axs[i], label='KDE')
    
    # Calculate the mean and median
    mean = data['flipper_length_mm'].mean()
    median = data['flipper_length_mm'].median()
    
    # Add vertical lines for the mean and median
    axs[i].axvline(mean, color='red', linestyle='--', label='Mean')
    axs[i].axvline(median, color='green', linestyle='-', label='Median')
    
    # Set the title and labels
    axs[i].set_title(f'Species: {species}')
    axs[i].set_xlabel('Flipper Length (mm)')
    axs[i].set_ylabel('Density')
    axs[i].legend()

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()


# In[3]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the penguins dataset
penguins = sns.load_dataset('penguins')

# Create the figure and axes
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Iterate over each species
for i, (species, data) in enumerate(penguins.groupby('species')):
    # Create the KDE plot
    sns.kdeplot(data['flipper_length_mm'], ax=axs[i], label='KDE')
    
    # Calculate the descriptive statistics
    mean = data['flipper_length_mm'].mean()
    median = data['flipper_length_mm'].median()
    std = data['flipper_length_mm'].std()
    q1 = data['flipper_length_mm'].quantile(0.25)
    q3 = data['flipper_length_mm'].quantile(0.75)
    min_val = data['flipper_length_mm'].min()
    max_val = data['flipper_length_mm'].max()

    # Add vertical lines for mean and median
    axs[i].axvline(mean, color='red', linestyle='--', label='Mean')
    axs[i].axvline(median, color='green', linestyle='-', label='Median')
    
    # Shade the range
    axs[i].axhspan(0, 0.02, xmin=(min_val-150)/100, xmax=(max_val-150)/100, color='blue', alpha=0.2, label='Range')

    # Shade the IQR
    axs[i].axhspan(0, 0.015, xmin=(q1-150)/100, xmax=(q3-150)/100, color='green', alpha=0.3, label='IQR')

    # Shade two standard deviations from the mean
    axs[i].axhspan(0, 0.01, xmin=(mean - 2*std - 150)/100, xmax=(mean + 2*std - 150)/100, color='gray', alpha=0.4, label='2 STD')
    
    # Set the title and labels
    axs[i].set_title(f'Species: {species}')
    axs[i].set_xlabel('Flipper Length (mm)')
    axs[i].set_ylabel('Density')
    axs[i].legend()

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()


# In[ ]:


Let's examine the strengths of histograms, particularly for large datasets, and consider the trade-offs compared to KDEs.

You're right to highlight the effectiveness of histograms in representing the shape of a distribution, especially when dealing with large datasets. Histograms excel in providing a clear and intuitive visualization of how data points are distributed across different value ranges. By grouping data into bins and representing the frequency within each bin as the height of a bar, histograms readily convey information about central tendency, spread, and potential skewness.

*   **Clarity and Intuition:** A key advantage of histograms, as you pointed out, is their clarity and ease of interpretation. The visual representation directly maps to the frequency of data points within specific ranges. This makes histograms particularly accessible, even for individuals who might be less familiar with statistical visualizations. 

*   **Handling Large Datasets:** Histograms are particularly well-suited for large datasets due to their ability to effectively summarize and visualize the distribution of a substantial number of data points.  The binning process helps in condensing the information, making it easier to grasp the overall pattern within the data without getting overwhelmed by individual data points.

However, it's important to acknowledge that the choice of bin size in a histogram can influence its appearance and interpretation.  While a well-chosen bin size can reveal the underlying shape of the distribution, an inappropriate bin size might either oversimplify or overcomplicate the representation.

*   **Impact of Bin Size:** If the bin size is too small, the histogram might become excessively granular, potentially obscuring the overall pattern with noise. Conversely, if the bin size is too large, the histogram might over-smooth the data, potentially masking important details like multimodality.

In contrast to histograms, which use discrete bins, KDEs employ a kernel function to create a smooth, continuous estimate of the probability density function. While KDEs can provide a visually appealing and potentially more nuanced representation of the distribution, they might require a deeper understanding of statistical concepts for accurate interpretation, particularly regarding the density values on the y-axis.

The choice between a histogram and a KDE often depends on the specific goals of the analysis and the familiarity of the audience with statistical visualization techniques. 

You've correctly identified the strengths of histograms, especially their clarity and suitability for large datasets. However, it's valuable to be mindful of the potential impact of bin size choices and consider KDEs when a smoother representation or a different perspective on the probability density might be beneficial.


# In[4]:


from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

n = 1500
data1 = stats.uniform.rvs(0, 10, size=n)
data2 = stats.norm.rvs(5, 1.5, size=n)
data3 = np.r_[stats.norm.rvs(2, 0.25, size=int(n/2)), stats.norm.rvs(8, 0.5, size=int(n/2))]
data4 = stats.norm.rvs(6, 0.5, size=n)

fig = make_subplots(rows=1, cols=4)

fig.add_trace(go.Histogram(x=data1, name='A', nbinsx=30, marker=dict(line=dict(color='black', width=1))), row=1, col=1)
fig.add_trace(go.Histogram(x=data2, name='B', nbinsx=15, marker=dict(line=dict(color='black', width=1))), row=1, col=2)
fig.add_trace(go.Histogram(x=data3, name='C', nbinsx=45, marker=dict(line=dict(color='black', width=1))), row=1, col=3)
fig.add_trace(go.Histogram(x=data4, name='D', nbinsx=15, marker=dict(line=dict(color='black', width=1))), row=1, col=4)

fig.update_layout(height=300, width=750, title_text="Row of Histograms")
fig.update_xaxes(title_text="A", row=1, col=1)
fig.update_xaxes(title_text="B", row=1, col=2)
fig.update_xaxes(title_text="C", row=1, col=3)
fig.update_xaxes(title_text="D", row=1, col=4)
fig.update_xaxes(range=[-0.5, 10.5])

for trace in fig.data:
    trace.xbins = dict(start=0, end=10)
    
# This code was produced by just making requests to Microsoft Copilot
# https://github.com/pointOfive/stat130chat130/blob/main/CHATLOG/wk3/COP/SLS/0001_concise_makeAplotV1.md

fig.show() # USE `fig.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS


# In[ ]:


# Datasets B and D likely have similar variances, as their standard deviations (1.5 and 0.5, respectively) are relatively close. Dataset A also might appear to have a similar variance depending on the random values generated. However, it's crucial to remember that Dataset A is a uniform distribution, inherently different from the normal distributions of Datasets B and D. 
# Datasets B and D have different means (5 and 6, respectively) but could be considered to have different variances depending on the randomly generated values. Dataset A might appear to have a similar mean to Datasets B and D depending on the random values generated, but again, it's important to note the difference in distribution type.
#Datasets B and D might appear to have similar variances depending on the random values generated, but their means (5 and 6) are different.
#Dataset C has two distinct peaks, indicating a different distribution shape and, therefore, potentially different means and variances compared to the other datasets.


# In[7]:


from scipy import stats
import pandas as pd
import numpy as np
  
sample1 = stats.gamma(a=2,scale=2).rvs(size=1000)
fig1 = px.histogram(pd.DataFrame({'data': sample1}), x="data")
# USE `fig1.show(renderer="png")` FOR ALL GitHub and MarkUs SUBMISSIONS

sample1.mean()
np.quantile(sample1, [0.5]) # median

sample2 = -stats.gamma(a=2,scale=2).rvs(size=1000)
# when it is symmetrical distribution, the means is euqal to the median. When it is the right skewwed distribution, the mean is greater than median. When it is the left skewwed distribution, the median is greater than means.


# In[ ]:


Let's examine the code and break down its purpose:

*   **Generating Sample Data**

    *   `sample1 = stats.gamma(a=2, scale=2).rvs(size=1000)`: This line generates 1000 random data points from a gamma distribution. 

        *   The gamma distribution is a flexible distribution often used to model positive, right-skewed data, such as waiting times or income distributions. 
        *   The `a=2` argument refers to the shape parameter, and `scale=2` refers to the scale parameter of the gamma distribution. These parameters control the shape and spread of the distribution.
        *   `rvs(size=1000)` generates 1000 random variates (data points) from this specific gamma distribution.
    *   `sample2 = -stats.gamma(a=2,scale=2).rvs(size=1000)`: This line generates another 1000 random data points from the same gamma distribution but multiplies each value by -1. This has the effect of reflecting the distribution horizontally, effectively creating a left-skewed dataset from the original right-skewed gamma distribution. This is likely done to demonstrate the impact of skewness on the relationship between the mean and median. 
*   **Visualizing the Data:**

    *   `fig1 = px.histogram(pd.DataFrame({'data': sample1}), x="data")`: This line creates a histogram using the plotly library (`px`) to visualize the distribution of `sample1`. 

        *   Histograms are useful for visualizing the distribution of numerical data by dividing the data into bins and displaying the frequency (or count) of data points within each bin.
        *   The code assumes you have the plotly.express library imported as `px` (although this import is not explicitly provided in your current code snippet).
*   **Calculating Mean and Median**

    *   `sample1.mean()`: This calculates the **mean** (average) of the data points in `sample1`.
    *   `np.quantile(sample1, [0.5]) # median`: This calculates the 50th percentile (the **median**) of the `sample1` data.

        *   The median is the middle value when the data is ordered.
        *   The comment in the code clarifies that this line is calculating the median.

**Purpose of the Code**

The primary goal of this code is likely to demonstrate how skewness influences the relationship between the **mean** and **median**. 

*   In a right-skewed distribution (like the gamma distribution used here), the mean is typically greater than the median. This is because the mean is influenced by extreme values (outliers) in the right tail, while the median is less affected.
*   By reflecting the gamma distribution (multiplying by -1), the code creates a left-skewed distribution. In this case, you would observe the opposite: the mean would be less than the median because the mean is pulled towards the outliers in the left tail.

The code allows you to visually compare the histograms of a right-skewed and a left-skewed distribution while also examining the numerical values of the mean and median for each. This comparison helps illustrate how the relationship between the mean and median changes with skewness.


# In[8]:


import pandas as pd
import plotly.express as px

# Read the data from the URL
url = "https://raw.githubusercontent.com/steview-d/superhero-dashboard/master/static/data/superheroData.csv"
superhero_df = pd.read_csv(url)

# Display the first few rows of the DataFrame to verify
print(superhero_df.head()) 

# --- Histogram ---
fig_histogram = px.histogram(superhero_df, x='Height')
fig_histogram.show()

# --- Box Plot ---
fig_boxplot = px.box(superhero_df, y='Weight')
fig_boxplot.show()

# --- Scatter Plot ---
fig_scatter = px.scatter(superhero_df, x='Intelligence', y='Combat')
fig_scatter.show()


# In[ ]:


#Just somesome. I will try my best to learn this courses. It is really hard and challenging.

