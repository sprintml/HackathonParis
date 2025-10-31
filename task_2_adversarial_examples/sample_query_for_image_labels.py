import numpy as np
import requests

# This Code can be used to query the model during adversarial example generation

BASE_URL = "http://34.122.51.94:9000"

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
output_path = "./query_images.npz"
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