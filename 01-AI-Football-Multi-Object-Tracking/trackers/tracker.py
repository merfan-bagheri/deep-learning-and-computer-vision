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

class Tracker:
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
            "players":[],
            "referees":[],
            "ball":[]
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
                cls_names = detection.names
                cls_names_inv = {v:k for k,v in cls_names.items()}
                
                
                # Covert to supervision Detection format
                detection_supervision = sv.Detections.from_ultralytics(detection)
                
                # Track Objects
                detection_with_tracks = bytetrack_tracker.update_with_detections(detection_supervision)
                
                tracks["players"].append({})
                tracks["referees"].append({})
                tracks["ball"].append({})
                
                for frame_detection in detection_with_tracks:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]
                    track_id = frame_detection[4]
                    
                    if cls_id == cls_names_inv['player']:
                        tracks["players"][frame_num][track_id] = {"bbox":bbox}
                        bbox = [round(value) for value in xyxy_to_xywh(bbox)]
                        line = f"{frame_num+1}, {track_id}, {bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}\n"
                        f.write(line)
                    
                    elif cls_id == cls_names_inv['referee']:
                        tracks["referees"][frame_num][track_id] = {"bbox":bbox}
                
                for frame_detection in detection_supervision:
                    bbox = frame_detection[0].tolist()
                    cls_id = frame_detection[3]

                    if cls_id == cls_names_inv['ball']:
                        tracks["ball"][frame_num][1] = {"bbox":bbox}
        print(f"ByteTrack tracking data saved to {output_path}")
        return tracks
        
    def add_position_to_tracks(sekf,tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
    
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

    def draw_annotations(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            # Draw Players
            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))

            # Draw Referee
            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            # Draw Ball Trajectory if enabled
            if self.draw_trajectory:
                tail_positions = []

                # Determine the range of frames to consider for the trajectory
                start_frame = max(0, frame_num - self.trajectory_length + 1)
                end_frame = frame_num + 1  # Include current frame

                # Collect positions of the ball over the last n frames
                for i in range(start_frame, end_frame):
                    ball_tracks = tracks["ball"][i]
                    if 1 in ball_tracks and 'position' in ball_tracks[1]:
                        pos = ball_tracks[1]['position']
                        tail_positions.append(pos)

                # Interpolate positions to create smoother trajectory
                interpolated_positions = []
                if len(tail_positions) >= 2:
                    # Number of interpolated points between each pair of positions
                    num_interpolated_points = 5

                    for idx in range(len(tail_positions) - 1):
                        start_pos = np.array(tail_positions[idx])
                        end_pos = np.array(tail_positions[idx + 1])

                        # Linearly interpolate between start_pos and end_pos
                        for t in np.linspace(0, 1, num_interpolated_points, endpoint=False):
                            interp_pos = (1 - t) * start_pos + t * end_pos
                            interpolated_positions.append(interp_pos)

                    # Add the last position
                    interpolated_positions.append(tail_positions[-1])
                elif tail_positions:
                    # If only one position is available
                    interpolated_positions = tail_positions.copy()

                # Create an overlay for transparent drawing
                overlay = frame.copy()

                num_positions = len(interpolated_positions)
                for idx, pos in enumerate(interpolated_positions):
                    x, y = int(pos[0]), int(pos[1])

                    # Adjust transparency based on index (older positions are more transparent)
                    alpha = (idx + 1) / num_positions  # From 1/num_positions to 1
                    alpha = alpha * 0.4  # Scale alpha (max 0.8 to avoid full opacity)

                    # Sky blue color in BGR
                    color = (235, 206, 135)
                    # color = (255, 0, 0)

                    # Draw circle on overlay
                    cv2.circle(overlay, (x, y), radius=2, color=color, thickness=-1)

                    # Blend overlay with original frame based on alpha
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                    # Reset overlay for the next circle
                    overlay = frame.copy()

            # Draw Ball (Optional: If you still want to highlight the current ball position)
            for _, ball in ball_dict.items():
                frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))

            # Append the annotated frame to the output list
            output_video_frames.append(frame)

        return output_video_frames

