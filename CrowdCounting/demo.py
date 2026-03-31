import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torchvision.transforms.functional as TF
import random
import numpy as np
from PIL import Image
import math
import scipy.io as sio
import scipy.ndimage as ndimage
import wandb
import matplotlib.pyplot as plt
import cv2
from huggingface_hub import snapshot_download

from model import CSRNet
torch.backends.cudnn.benchmark = True

def run_video_demo(model, video_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    print("Video processing started! Press 'q' to stop.")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached.")
                break

            img_resized = cv2.resize(frame, (432, 240))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_tensor = TF.to_tensor(img_rgb)
            img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            img_tensor = img_tensor.unsqueeze(0).to(device)

            output = model(img_tensor)
            output = torch.relu(output)
            pred_count = torch.sum(output).item()

            output = func.interpolate(
                output, 
                size=(img_tensor.size(2), img_tensor.size(3)), 
                mode='bilinear', 
                align_corners=False
            )
            
            density_map = output.cpu().squeeze().numpy()
            heatmap_norm = cv2.normalize(density_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            heatmap_norm = np.uint8(heatmap_norm)
            heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
            blended = cv2.addWeighted(img_resized, 0.6, heatmap_color, 0.4, 0)
            display_frame = cv2.resize(blended, (864, 480))

            cv2.putText(display_frame, f"Estimated Students: {pred_count:.1f}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.imshow('Accurate CSRNet Video Demo', display_frame)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def find_available_cameras(max_to_test=5):
    """Scans the PC for connected cameras and returns their indexes."""
    print("Scanning USB ports and built-in webcams...")
    available_cameras = []
    
    for i in range(max_to_test):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) 
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
            
    return available_cameras

def run_selectable_camera_demo(model, camera_index):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {camera_index}.")
        return

    print(f"\nLive feed started on Camera {camera_index}! Press 'q' on the video window to exit.")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            img_resized = cv2.resize(frame, (432, 240))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_tensor = TF.to_tensor(img_rgb)
            img_tensor = TF.normalize(img_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            img_tensor = img_tensor.unsqueeze(0).to(device)

            output = model(img_tensor)
            output = torch.relu(output)
            pred_count = torch.sum(output).item()
            output_upscaled = func.interpolate(
                output, 
                size=(240, 432), 
                mode='bilinear', 
                align_corners=False
            )

            density_map = output_upscaled.cpu().squeeze().numpy()
            heatmap_norm = cv2.normalize(density_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            heatmap_norm = np.uint8(heatmap_norm)
            heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
            blended = cv2.addWeighted(img_resized, 0.6, heatmap_color, 0.4, 0)
            display_frame = cv2.resize(blended, (864, 480))

            cv2.putText(display_frame, f"Live Crowd Count: {pred_count:.1f}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            cv2.imshow(f'CSRNet Demo - Camera {camera_index}', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

checkpoint_path = "./checkpoints/"
checkpoint_name = "xinnguoihayvenoiday105.pth"

snapshot_download(
    repo_id="b1nswing/CSRNET_config_B", 
    local_dir="./checkpoints"
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Evaluating on {device}...")

model = CSRNet().to(device)

checkpoint = torch.load(checkpoint_path+checkpoint_name, map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])

# target_video = "./shanghaitech/training/videos/01_002.avi" 
# run_video_demo(model, target_video)
cameras = find_available_cameras()
    
if not cameras:
    print("No cameras detected on this system!")
else:
    print("\nAvailable Cameras found at indexes:", cameras)
    try:
        selected_cam = int(input(f"Enter the camera index you want to use {cameras}: "))
        if selected_cam in cameras:
            run_selectable_camera_demo(model, selected_cam)
        else:
            print("Invalid camera index selected.")
    except ValueError:
        print("Please enter a valid number.")