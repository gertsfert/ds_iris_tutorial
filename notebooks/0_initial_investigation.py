# %% [markdown]
# ## Initial Investigations
#
# Doing some basic investigations into the dataset

# %% Initial Imports
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

# loading data
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv('data/raw/iris.csv', names=names)

# %% [markdown]
# ## Basic Dataset Investigation

# %% [markdown]
# ### Shape
print(f'(rows, columns): {dataset.shape}')

# %% [markdown]
# ### Head
dataset.head()

# %% [markdown]
# ### Statistical Summaries
dataset.describe()

# %% [markdown]
# ### Class Distribution
dataset.groupby('class').size()

# %% [markdown]
# ## Investigative Plotting - Singlevariate

# %% [markdown]
# ### Boxplot
dataset.plot(kind='box', subplots=True,
             layout=(2, 2), sharex=False, sharey=False)
plt.show()

# %% [markdown]
# ### Histogram
dataset.hist()
plt.show()

# %% [markdown]
# ## Investigative Plotting - Multivariate

# %% [markdown]
# ### Scatter plot matrix
scatter_matrix(dataset)
plt.show()

# %% Save as interim
dataset.to_csv(r'data/interim/iris_labelled.csv', index=False)


# %%
