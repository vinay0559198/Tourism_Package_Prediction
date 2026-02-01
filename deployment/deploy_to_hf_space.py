from huggingface_hub import HfApi

SPACE_REPO = "vinay9700/tourism-package-prediction-space"

api = HfApi()

# Create Space (Docker-based)
api.create_repo(
    repo_id=SPACE_REPO,
    repo_type="space",
    space_sdk="docker",
    private=False,
    exist_ok=True
)

# Upload deployment files
files = ["Dockerfile", "app.py", "requirements.txt"]

for file in files:
    api.upload_file(
        path_or_fileobj=file,
        path_in_repo=file,
        repo_id=SPACE_REPO,
        repo_type="space"
    )

print("✅ Deployment files pushed to Hugging Face Space")
