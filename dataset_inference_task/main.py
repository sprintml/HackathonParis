import torchvision.models as models
import torch as th
import pandas as pd
import numpy as np
import requests

model = models.resnet18(weights=None)
model.conv1 = th.nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=3, bias=False)
model.fc = th.nn.Sequential(
    th.nn.Dropout(p=0.2),
    th.nn.Linear(model.fc.in_features, 14)
)
device = th.device("cuda" if th.cuda.is_available() else "cpu")

model.load_state_dict(th.load("classifier.pt", map_location="cpu"))
model.to(device)

# Load the dataset
dataset = th.load("subsets_dataset.pt")

# Example: How to access a subset
subset_0 = dataset["subset_0"]
print(subset_0)
subset_999 = dataset["subset_999"]
print(subset_999)

images = [img.flatten().tolist() for img in subset_0["images"]]

TOKEN = "REPLACE-WITH-YOUR-TOKEN"

# Sample submission
df = pd.DataFrame(
    {
        "ids": list(range(len(dataset))),
        "score": np.random.randn(len(dataset)),
    }
)
df.to_csv("test.csv", index=None)

response = requests.post("http://34.122.51.94:9090/06-dataset-inference-vision", files={"file": open("test.csv", "rb")}, headers={"token": TOKEN})
print(response.json())