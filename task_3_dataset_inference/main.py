import torch as th
import pandas as pd
import requests
import time
import sys
import torchvision.models as models

# --------------------------------
# DATASET
# --------------------------------

"""
Dataset contents:

- 1000 subsets of images, each subset stored under the key "subset_{i}" where i ranges from 0 to 999.
  Each subset is a dictionary with:
    -"images": Tensor containing the 100 images in the subset, has shape (100, 1, 28, 28)
    -"labels": Tensor of true labels for the images in the subset, has shape (100)
    -"subset_id": Integer ID of the subset (from 0 to 999)
"""

# Load the dataset
dataset = th.load("subsets_dataset.pt")

# Example: Acessing subsets
subset_0 = dataset["subset_0"]

print("Subset 0 keys:", subset_0.keys())
print("Subset ID:", subset_0["subset_id"])
print("Images shape:", subset_0["images"].shape)
print("Labels shape:", subset_0["labels"].shape) 
print("First image tensor:", subset_0["images"][:1])
print("First 10 labels:", subset_0["labels"][:10])

# --------------------------------
# QUERYING THE CLASSIFIER
# --------------------------------

# You can use the following Code to load and query the image classifier with images:

model = models.resnet18(weights=None)
model.conv1 = th.nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=3, bias=False)
model.fc = th.nn.Sequential(
    th.nn.Dropout(p=0.2),
    th.nn.Linear(model.fc.in_features, 9)
)
device = th.device("cuda" if th.cuda.is_available() else "cpu")

model.load_state_dict(th.load("classifier.pt", map_location="cpu"))
model.to(device)

images = subset_0["images"]

output = model(images)

print("Logits shape:", output.shape)  # Should be (100, 9) for subset of 100 images and 9 classes
print("First 5 logits:", output[:5]) 

# --------------------------------
# SUBMISSION FORMAT
# --------------------------------

"""
The submission must be a .csv file with the following format:

-"subset_id": ID of the subset (from 0 to 999)
-"membership": Membership score for each image (float)
"""

# Example Submission:

subset_ids = list(range(len(dataset)))  
membership_scores = th.rand(len(dataset)).tolist()
submission_df = pd.DataFrame({
    "subset_id": subset_ids,
    "membership": membership_scores
})
submission_df.to_csv("example_submission.csv", index=None)

# --------------------------------
# SUBMISSION PROCESS
# --------------------------------

BASE_URL = "http://34.122.51.94:9000"
API_KEY  = "INSERT_YOUR_API_KEY_HERE" 

TASK_ID  = "06-dataset-inference-vision"
FILE_PATH = "example_submission.csv" # <- Path to your real submission file

try:
    with open(FILE_PATH, "rb") as f:
        response = requests.post(
            f"{BASE_URL}/submit/{TASK_ID}",
            headers={"X-API-Key": API_KEY},
            files={"file": f},
            timeout=60,
        )

    response.raise_for_status()
    submission_data = response.json()
    submission_id = submission_data.get("submission_id")

    print(f"✅ Successfully submitted. Submission ID: {submission_id}")
    print("Initial response:", submission_data)

except requests.exceptions.RequestException as e:
    print(f"❌ An error occurred during submission: {e}")
    if e.response is not None:
        print("Response body:", e.response.text)
    sys.exit(1)

while True:
    try:
        time.sleep(10)
        status_response = requests.get(
            f"{BASE_URL}/submissions/{submission_id}",
            headers={"X-API-Key": API_KEY},
        )
        status_response.raise_for_status()
        status_data = status_response.json()
        status = status_data.get("status")

        print(f"Submission Status: {status}")

        if status in ["done", "failed"]:
            print("Final submission data:", status_data)
            break

    except requests.exceptions.RequestException as e:
        print(f"❌ An error occurred while checking submission status: {e}")
        if e.response is not None:
            print("Response body:", e.response.text)
        sys.exit(1)