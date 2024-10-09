# Python Libraries for Data Science

This repository provides an introduction to key Python libraries used in data science, including their installation and basic usage. It covers libraries such as `NumPy`, `pandas`, `Matplotlib`, `Scikit-learn`, `TensorFlow`, and tools for web scraping. Each section includes installation commands, basic commands, and a simple example to help you get started.

## Table of Contents
- [Installation](#installation)
- [Libraries Overview](#libraries-overview)
  - [NumPy](#numpy)
  - [pandas](#pandas)
  - [Matplotlib](#matplotlib)
  - [Web Scraping Tools](#web-scraping-tools)
  - [Scikit-learn](#scikit-learn)
  - [TensorFlow](#tensorflow)
- [Contributing](#contributing)
- [License](#license)

## Installation

You can install the required libraries using `pip`. Run the following command:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow beautifulsoup4 requests
```

This will install:
- NumPy for numerical computing
- pandas for data manipulation
- Matplotlib for data visualization
- scikit-learn for machine learning
- TensorFlow for deep learning
- BeautifulSoup and requests for web scraping

## Libraries Overview

### NumPy
NumPy is the fundamental package for numerical computing in Python. It provides support for arrays, matrices, and many mathematical functions.

#### Installation
```bash
pip install numpy
```

#### Basic Usage
```python
import numpy as np

# Create an array
arr = np.array([1, 2, 3])
print(arr)

# Perform operations
print(np.mean(arr))
```

### pandas
pandas is an open-source library that provides high-performance data manipulation and analysis tools, particularly DataFrames.

#### Installation
```bash
pip install pandas
```

#### Basic Usage
```python
import pandas as pd

# Create a DataFrame
data = {'Name': ['Alice', 'Bob'], 'Age': [25, 30]}
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

# Perform basic operations
print(df.describe())
```

### Matplotlib
Matplotlib is a plotting library used for creating static, interactive, and animated visualizations in Python.

#### Installation
```bash
pip install matplotlib
```

#### Basic Usage
```python
import matplotlib.pyplot as plt

# Create a simple plot
plt.plot([1, 2, 3], [4, 5, 6])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Plot')
plt.show()
```

### Web Scraping Tools
BeautifulSoup and requests are essential libraries for web scraping, allowing you to extract data from websites.

#### Installation
```bash
pip install beautifulsoup4 requests
```

#### Basic Usage
```python
import requests
from bs4 import BeautifulSoup

# Fetch content from a webpage
url = 'https://example.com'
response = requests.get(url)

# Parse HTML content
soup = BeautifulSoup(response.text, 'html.parser')
print(soup.title.text)
```

### Scikit-learn
Scikit-learn is a library for machine learning, offering simple and efficient tools for data analysis and modeling.

#### Installation
```bash
pip install scikit-learn
```

#### Basic Usage
```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Sample data
X = np.array([[1], [2], [3]])
y = np.array([2, 4, 6])

# Create a model and fit it
model = LinearRegression()
model.fit(X, y)

# Make a prediction
print(model.predict([[4]]))
```

### TensorFlow
TensorFlow is an open-source platform for machine learning and deep learning, commonly used for building neural networks.

#### Installation
```bash
pip install tensorflow
```

#### Basic Usage
```python
import tensorflow as tf

# Create a constant tensor
hello = tf.constant('Hello, TensorFlow!')
print(hello.numpy())
```

## Contributing
If you'd like to contribute, feel free to fork the repository and submit a pull request. For major changes, please open an issue to discuss what you would like to change.

## License
This project is licensed under the MIT License.
