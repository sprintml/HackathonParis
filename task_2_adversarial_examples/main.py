import torch as th
import requests
import time
import sys

# --------------------------------
# DATASET
# --------------------------------

"""
Dataset contents:

-"images": Tensor containing the 1,000 natural images, has shape (1000, 3, 28, 28)
-"labels": Tensor of true labels for the images, has shape (1000, 1)
"""

# Load the dataset
dataset = th.load("natural_images.pt")

# --------------------------------
# QUERYING THE CLASSIFIER
# --------------------------------

# You can use the following Code to query the image classifier with images, and get back the corresponding logits:

BASE_URL = "http://34.122.51.94:9000"

url = f"{BASE_URL}/10-adversarial-examples/logits"

N = 100  # Number of sample images to query

# Generate sample images
images = th.randint(0, 256, size=(N, 3, 28, 28), dtype=th.uint8) # <- Insert your input images here
# Normalize to [0, 1] range
images = images.float() / 255.0

# Generate sequential image IDs
image_ids = th.arange(N, dtype=th.int32)

# Save both images and their IDs
output_path = "./query_images.pt"
th.save({
    "images": images,
    "image_ids": image_ids
}, output_path)

# Query the model
with open(output_path, "rb") as f:
    response = requests.post(
        url,
        files={"pt": f},
        timeout=60,
    )

if response.status_code == 200:
    data = response.json()
    print("✅ Request successful")
    print(f"Batch size: {data['batch_size']}")
    print(f"Num classes: {data['num_classes']}")
    print("\nSample of results:")
    # Print first few results to show the structure
    for result in data['results'][:5]:  # Show first 5 results
        print(f"Image ID: {result['image_id']}")
        print(f"Logits: {result['logits']}\n")
else:
    print("❌ Request failed")
    print("Status code:", response.status_code)
    try:
        print("Error message:", response.json())
    except:
        print("Error message:", response.text)


# --------------------------------
# SUBMISSION FORMAT
# --------------------------------

"""
The submission must be a .pt file of the following format:

-"images": Tensor containing the generated adversarial examples in the same order as the corresponding
           natural images, has shape (N, 3, 28, 28)
"""

# Example Submission:

adversarial_examples = th.randint(0, 256, size=(1000, 3, 28, 28), dtype=th.uint8)
th.save({'images': adversarial_examples}, "example_submission.pt")

# --------------------------------
# SUBMISSION PROCESS
# --------------------------------

# You can submit your adversarial examples using the following code:

API_KEY  = "INSERT_YOUR_API_KEY_HERE"

TASK_ID  = "10-adversarial-examples"
FILE_PATH = "example_submission.pt" # <- Path to your real submission file

# Submit the solutions
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

