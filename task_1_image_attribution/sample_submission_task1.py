import os
import requests
import time
import json
import sys


BASE_URL = "http://34.122.51.94:9000"
API_KEY  = "YOUR_API_KEY" 

TASK_ID  = "05-iar-attribution"         
FILE_PATH = "PATH/TO/YOUR/SUBMISSION_FILE.csv" 

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