import torch as th
import requests
import sys
import os
import numpy as np

# --------------------------------
# DATASET
# --------------------------------

"""
Dataset contents:

-"images": Tensor containing the 1,000 natural images, has shape (1000, 3, 28, 28)
-"labels": Tensor of true labels for the images, has shape (1000)
"""

# Load the dataset
dataset = th.load("natural_images.pt")

print("Dataset keys:", dataset.keys())
print("Images shape:", dataset["images"].shape)
print("Labels shape:", dataset["labels"].shape)
print("First 10 labels:", dataset["labels"][:10])
print("First image tensor:", dataset["images"][:1])

# --------------------------------
# QUERYING THE CLASSIFIER
# --------------------------------

# You can use the following Code to query the image classifier with images, and get back the corresponding logits:

BASE_URL = "http://34.122.51.94:9000"

url = f"{BASE_URL}/10-adversarial-examples/logits"

N = 100  # Number of sample images to query

# Generate sample images
images = th.randint(0, 256, size=(N, 3, 28, 28), dtype=th.uint8) # <- Insert your input images here
images = images.float() / 255.0 # Normalize to [0, 1] range (if they are not already)

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
The submission must be a .npz file of the following format:

-"images": Tensor containing the generated adversarial examples in the same order as the corresponding
           natural images, has shape (N, 3, 28, 28)
"""

# Example Submission:

adversarial_examples = th.randint(0, 256, size=(1000, 3, 28, 28), dtype=th.uint8)

adversarial_examples = adversarial_examples.float() / 255.0  # normalize to [0, 1] range

images_np = adversarial_examples.detach().cpu().numpy()

np.savez_compressed("example_submission.npz", images=images_np)

# --------------------------------
# SUBMISSION PROCESS
# --------------------------------

"""
Adversarial Examples Task — Participant Submission Guide
========================================================

You will upload a single **.npz** file that contains ONLY an array named **'images'**.
The evaluator will load your file, run shape/dtype checks against the natural images,
and then score it by running a fixed classifier and measuring perturbations.

Follow these rules carefully to avoid automatic rejection.

1) File format
--------------
- **Extension:** `.npz` (NumPy compressed archive)
- **Content:** must contain exactly one required key: `'images'`
- **Max file size:** 200 MB (hard limit). Larger files are rejected.

2) Array requirements
---------------------
Let `G` be the ground-truth tensor loaded:

- **Shape:** `images.shape` must match `G["images"].shape` **exactly**.
  - If `G["images"]` is `(N, 3, H, W)`, your array must also be `(N, 3, H, W)`.
  - No extra samples; no fewer; no different dimensions.
- **Dtype:** `images.dtype` must match `G["images"].dtype` **exactly**.
  - If the GT uses `float32`, you must submit `float32`.
  - Safe cast example: `images = np.asarray(images, dtype=np.float32)`
- **Finite values only:** No NaN or Inf anywhere.
  - The evaluator checks: `torch.isfinite(images).all()`.
- **Contiguity:** The server will convert to a contiguous Torch tensor; standard NumPy arrays are fine.


3) Typical failure messages & what they mean
--------------------------------------------
- "File must be .npz and contain an 'images' array."
  → Wrong extension or missing `'images'` key.
- "File too large: X bytes (limit 209715200)."
  → Your file exceeds 200 MB.
- "Failed to read .npz: ..."
  → The file is corrupted or not a valid `.npz` created with `allow_pickle=False`.
- "Failed to convert 'images' to torch tensor: ..."
  → Your `'images'` array has an unsupported dtype or structure (e.g., object array).
- "Submitted images must have shape (N, C, H, W), but got (...)."
  → Shape mismatch with the ground-truth images.
- "Submitted images must be of type torch.float32, but got torch.float64."
  → Dtype mismatch with the ground-truth images.
- "Images must not contain NaN or Inf values."
  → Clean your array: `np.isfinite(images).all()` must be True.
"""

API_KEY  = "YOUR_API_KEY_HERE"  

TASK_ID = "10-adversarial-examples"

# Path to the .npz file you want to send
FILE_PATH = "PATH/TO/YOUR/SUBMISSION.npz"

GET_LOGITS = False      # set True to get logits from the API
SUBMIT     = False      # set True to submit your solution
GET_STATUS = False      # set True to poll with a known submission_id

KNOWN_SUBMISSION_ID = "Your_Submission_ID_Here"  # paste a known ID here

def die(msg):
    print(f"{msg}", file=sys.stderr)
    sys.exit(1)

if GET_LOGITS:

    with open(FILE_PATH, "rb") as f:
        files = {"npz": (FILE_PATH, f, "application/octet-stream")}
        response = requests.post(
            f"{BASE_URL}/{TASK_ID}/logits", 
            files=files)

    if response.status_code == 200:
        data = response.json()
        print("Request successful")
        print(data)

    else:
        print("Request failed")
        print("Status code:", response.status_code)
        print("Detail:", response.text)

if SUBMIT:
    if not os.path.isfile(FILE_PATH):
        die(f"File not found: {FILE_PATH}")

    try:
        with open(FILE_PATH, "rb") as f:
            files = {
                "file": (os.path.basename(FILE_PATH), f, "csv"),
            }
            resp = requests.post(
                f"{BASE_URL}/submit/{TASK_ID}",
                headers={"X-API-Key": API_KEY},
                files=files,
                timeout=(10, 120), 
            )
        try:
            body = resp.json()
        except Exception:
            body = {"raw_text": resp.text}

        if resp.status_code == 413:
            die("Upload rejected: file too large (HTTP 413). Reduce size and try again.")

        resp.raise_for_status()

        submission_id = body.get("submission_id")
        print("Successfully submitted.")
        print("Server response:", body)
        if submission_id:
            print(f"Submission ID: {submission_id}")

    except requests.exceptions.RequestException as e:
        detail = getattr(e, "response", None)
        print(f"Submission error: {e}")
        if detail is not None:
            try:
                print("Server response:", detail.json())
            except Exception:
                print("Server response (text):", detail.text)
        sys.exit(1)

if GET_STATUS:

    submission_id = KNOWN_SUBMISSION_ID
    if not submission_id:
        die("Please set KNOWN_SUBMISSION_ID to check a submission’s status.")

    try:
        resp = requests.get(
            f"{BASE_URL}/submissions/{submission_id}",
            headers={"X-API-Key": API_KEY},
            timeout=(10, 30),
        )
        try:
            body = resp.json()
        except Exception:
            body = {"raw_text": resp.text}

        resp.raise_for_status()

        status = body.get("status")
        print(f"Submission Status: {status}")
        print("Full response:", body)

    except requests.exceptions.RequestException as e:
        detail = getattr(e, "response", None)
        print(f"Status check error: {e}")
        if detail is not None:
            try:
                print("Server response:", detail.json())
            except Exception:
                print("Server response (text):", detail.text)
        sys.exit(1)

