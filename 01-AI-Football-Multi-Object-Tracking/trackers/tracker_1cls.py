from ultralytics import YOLO
import supervision as sv
import pickle
import torch
import os
import numpy as np
import pandas as pd
import cv2
import sys 
from tqdm import tqdm
from IPython.display import clear_output
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position, xyxy_to_xywh

class Tracker_1cls:
    def __init__(self, model_path, input_frames_path, output_tracking_path, batch_size=16, trajectory_length=5, draw_trajectory=True):
        self.detections = None
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack()
        self.trajectory_length = trajectory_length
        self.draw_trajectory = draw_trajectory     
        self.if_path = input_frames_path
        self.ot_path = output_tracking_path
        self.batch_size = batch_size

    def detect_frames(self, confidence_level=0.1):
        self.detections = self.model.predict(self.if_path,
                                             conf=confidence_level, 
                                             stream=False, 
                                             batch=self.batch_size,
                                             # half=True,
                                             # vid_stride=2,
                                             verbose=False)
        return [self.detections]
        clear_output(wait=True)

    def bytetrack(self, 
                  track_activation_threshold=0.25,
                  lost_track_buffer=30,
                  minimum_matching_threshold=0.8,
                  frame_rate=30,
                  minimum_consecutive_frames=1):
        
        tracks={
            "players":[]
        }
        
        path = 'BT_tracker.txt'
        output_path = os.path.join(self.ot_path, path)
        bytetrack_tracker = sv.ByteTrack(track_activation_threshold=track_activation_threshold,
                                         lost_track_buffer=lost_track_buffer,
                                         minimum_matching_threshold=minimum_matching_threshold,
                                         minimum_consecutive_frames=minimum_consecutive_frames,
                                         frame_rate=frame_rate)
        
        with open(output_path, 'w') as f:
            for frame_num, detection in tqdm(enumerate(self.detections), total=len(self.detections), desc="ByteTrack Processing"):
                    
                # Covert to supervision Detection format
                detection_supervision = sv.Detections.from_ultralytics(detection)
                
                # Track Objects
                detection_with_tracks = bytetrack_tracker.update_with_detections(detection_supervision)
                
                tracks["players"].append({})
                
                for frame_detection in detection_with_tracks:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]
                    track_id = frame_detection[4]
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
                    bbox = [round(value) for value in xyxy_to_xywh(bbox)]
                    line = f"{frame_num+1}, {track_id}, {bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}\n"
                    f.write(line)
                        
        print(f"ByteTrack tracking data saved to {output_path}")
        return tracks
        
    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        lwidth = int(0.3*width)
        y_center = y2+lwidth

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), lwidth),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        if track_id is not None:
            
            rectangle_width = 16
            rectangle_height= 10
                        
            y_text = y_center + 4
            if track_id < 10:
                x_text = x_center-4
            elif track_id < 99:
                x_text = x_center-8
            else:
                x_text = x_center-10
                rectangle_width = 18
                rectangle_height= 10
            
            x1_rect = x_center - rectangle_width//2
            x2_rect = x_center + rectangle_width//2
            y1_rect = (y_center- rectangle_height//2) 
            y2_rect = (y_center+ rectangle_height//2)
            
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
                       
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x_text),int(y_text)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0,0,0),
                1
            )

        return frame

    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-5,y-10],
            [x+5,y-10],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 1)

        return frame

    def triangle(self, frame, bbox, color, track_id):
        y2 = int(bbox[3])-10
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        lwidth = int(0.2*width)
        y_center = y2+lwidth

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width*0.7), int(0.2*width)),
            angle=0.0,
            startAngle=0,
            endAngle=180,
            color = color,
            thickness=4,
            lineType=cv2.LINE_4
        )

        if track_id is not None:
            
            rectangle_width = 40
            rectangle_height= 20
            y_text = y_center + rectangle_height//4 - 1
            
            if track_id < 10:
                x_text = x_center - rectangle_width//4 + 5
            elif track_id < 99:
                x_text = x_center - rectangle_width//4 + 2
            else:
                rectangle_width = 48
                rectangle_height= 20
                x_text = x_center - rectangle_width//4 - 3
            
            x1_rect = x_center - rectangle_width//2
            x2_rect = x_center + rectangle_width//2
            y1_rect = (y_center- rectangle_height//2) 
            y2_rect = (y_center+ rectangle_height//2)
            
            
            
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            tx_width = 0.4         
            text_position = (int(x_text), int(y_text))

            # Define the border color and thickness
            border_color = (255, 255, 255)  # White border
            border_thickness = 3  # Thickness of the border

            cv2.putText(
                frame,
                f"{track_id}",
                (text_position[0] - 1, text_position[1] - 1),  # Slightly offset for the border
                cv2.FONT_HERSHEY_SIMPLEX,
                tx_width,
                border_color,
                border_thickness,
            )

            cv2.putText(
                frame,
                f"{track_id}",
                (text_position[0] + 1, text_position[1] - 1),  # Slightly offset for the border
                cv2.FONT_HERSHEY_SIMPLEX,
                tx_width,
                border_color,
                border_thickness,
            )

            cv2.putText(
                frame,
                f"{track_id}",
                (text_position[0] - 1, text_position[1] + 1),  # Slightly offset for the border
                cv2.FONT_HERSHEY_SIMPLEX,
                tx_width,
                border_color,
                border_thickness,
            )

            cv2.putText(
                frame,
                f"{track_id}",
                (text_position[0] + 1, text_position[1] + 1),  # Slightly offset for the border
                cv2.FONT_HERSHEY_SIMPLEX,
                tx_width,
                border_color,
                border_thickness,
            )

            # Finally, draw the main text on top
            cv2.putText(
                frame,
                f"{track_id}",
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                tx_width,
                (0, 0, 0),  # Main text color (black)
                1  # Thickness of main text
            )

        return frame
    
    def draw_annotations(self,video_frames, tracks, draw_type='ellipse'):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color",(0,0,255))
                if draw_type=='ellipse':
                    frame = self.draw_ellipse(frame, player["bbox"],color, track_id)
                elif draw_type=='tri':
                    frame = self.triangle(frame, player["bbox"],color, track_id)

            output_video_frames.append(frame)

        return output_video_frames