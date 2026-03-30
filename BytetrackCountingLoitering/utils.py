# utils.py
import cv2
import numpy as np
from config import polygon
import os
def inside_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, point, False) >= 0

def count_in_polygon(tracks, polygon):
    count = 0
    for track in tracks:
        x1, y1, x2, y2, track_id, class_id = track  # Unpack class_id
        if class_id == 0:  # Only count persons (class 0)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            if inside_polygon((cx, cy), polygon):
                count += 1
    return count
def gather_images_into_vid(image_folder, output_video_path, fps=30): 
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)]
    image_paths.sort()
    # lấy ra tấm ảnh đầu tiên 
    first_image = cv2.imread(image_paths[0])
    print("this is len", len(image_paths))
    height, width, _ = first_image.shape
    shape = (width, height)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, shape)
    
    for image_path in image_paths:
        img = cv2.imread(image_path)
        if img.shape[:2][::-1] == shape: 
            out.write(img)
    out.release()
    print("Video has been gathered")

if __name__ == "__main__":
    
    gather_images_into_vid("/home/minhtriet/Downloads/MOT20 (2)/train/MOT20-01/img1", "output.mp4", 20)