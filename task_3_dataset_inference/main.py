import torch as th
import pandas as pd
import numpy as np

# Load the dataset
dataset = th.load("subsets_dataset.pt")

# Example: How to access a subset
subset_0 = dataset["subset_0"]
print(subset_0)
subset_999 = dataset["subset_999"]
print(subset_999)

images_0 = [img.flatten().tolist() for img in subset_0["images"]]

TOKEN = "REPLACE-WITH-YOUR-TOKEN"

# Sample submission
df = pd.DataFrame(
    {
        "ids": list(range(len(dataset))),
        "score": np.random.randn(len(dataset)),
    }
)
df.to_csv("test.csv", index=None)

# Use the generated .csv file to evaluate your results using sample_submission_task3.py