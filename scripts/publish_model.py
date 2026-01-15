# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "huggingface_hub",
# ]
# ///
import argparse
from pathlib import Path
from huggingface_hub import HfApi


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload model to Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, required=True, help="Target Hugging Face Repo ID")
    parser.add_argument("--model_path", type=str, default="dist/holopaswin-v2", help="Local path to the model folder")
    parser.add_argument("--private", action="store_true", help="Make the repository private")
    parser.add_argument("--token", type=str, help="Hugging Face token")

    args = parser.parse_args()

    if not Path(args.model_path).exists():
        print(f"Error: Model path '{args.model_path}' does not exist.")
        return

    api = HfApi(token=args.token)

    print(f"Creating repository {args.repo_id} (Model) ...")
    try:
        api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    except Exception as e:
        print(f"Note on repo creation: {e}")

    print(f"Uploading files from {args.model_path} to {args.repo_id}...")
    try:
        api.upload_folder(
            folder_path=args.model_path,
            repo_id=args.repo_id,
            repo_type="model",
            path_in_repo=".",
        )
        print("Upload complete! 🚀")
        print(f"View your model at: https://huggingface.co/{args.repo_id}")
    except Exception as e:
        print(f"Upload failed: {e}")


if __name__ == "__main__":
    main()
