from huggingface_hub import HfApi

api = HfApi()
user = api.whoami()
print(f"User: {user['name']}")

# Create the space
try:
    repo = api.create_repo(
        repo_id="Crossroad777/solotab",
        repo_type="space",
        space_sdk="docker",
        private=False
    )
    print(f"Space created: {repo}")
except Exception as e:
    print(f"Error or already exists: {e}")
