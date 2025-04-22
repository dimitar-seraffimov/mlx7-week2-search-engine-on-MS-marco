from huggingface_hub import create_repo, upload_folder
from pathlib import Path

folder_path = Path(__file__).parent.parent / "dataprep" / "combined_vocab"

# my Hugging Face repo name
repo_id = "madnexx/ms-marco-tokenised"
# create the dataset repo (only runs if it doesn't exist already)
create_repo(repo_id, repo_type="dataset", exist_ok=True)

# upload the folder
upload_folder(
    repo_id=repo_id,
    folder_path=folder_path,
    repo_type="dataset",
    path_in_repo="",  # upload directly into root of repo
)