import cv2
import os
import natsort

def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30, 
                          (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()
    
def read_video(video_path):
    frames = []
    total_frames = 0
    if os.path.isfile(video_path):
        cap = cv2.VideoCapture(video_path) 
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in range(total_frames):
            ret, frame = cap.read()
            if ret:
                frames.append(frame) 
    else:
        image_files = [f for f in os.listdir(video_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        image_files = natsort.natsorted(image_files)
        image_paths = [os.path.join(video_path, f) for f in image_files]
        total_frames = int(len(image_files))
        for image_path in image_paths:
            frames.append(cv2.imread(image_path))
            
    return frames
