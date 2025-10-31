import requests
import time
import sys
import numpy as np


BASE_URL = "http://34.122.51.94:9000"
API_KEY  = "YOUR_API_KEY" 

TASK_ID  = "10-adversarial-examples"
FILE_PATH = "PATH/TO/YOUR/SUBMISSION_FILE.npz"

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

        if status in ["completed", "failed"]:
            print("Final submission data:", status_data)
            break

    except requests.exceptions.RequestException as e:
        print(f"❌ An error occurred while checking submission status: {e}")
        if e.response is not None:
            print("Response body:", e.response.text)
        sys.exit(1)

# Query the model for logits
url = f"{BASE_URL}/10-adversarial-examples/logits"  # Changed hyphen to underscore

N = 100  # Number of sample images to query

# Generate sample images
images = np.random.randint(0, 256, size=(N, 28, 28, 3), dtype=np.uint8)
# Normalize to [0, 1] range
images = images.astype(np.float32) / 255.0

# Generate sequential image IDs
image_ids = np.arange(N, dtype=np.int32)

# Save both images and their IDs
output_path = "./tasks/10_adversarial_examples/query_images.npz"
np.savez(
    output_path,
    images=images,
    image_ids=image_ids,
    allow_pickle=False  # Make sure to set allow_pickle to False
)

# Query the model
with open(output_path, "rb") as f:
    response = requests.post(
        url,
        files={"npz": f},
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