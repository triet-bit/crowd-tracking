from huggingface_hub import login, upload_folder
login()
upload_folder(folder_path="./checkpoints", repo_id="b1nswing/CSRNET_config_B", repo_type="model")