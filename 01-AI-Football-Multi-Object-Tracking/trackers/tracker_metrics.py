from ultralytics import YOLO
import supervision as sv
import numpy as np
import pandas as pd
import sys 
import cv2
import os
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
import natsort
from tqdm import tqdm
from IPython.display import clear_output
sys.path.append('../')
from deep_bytetrack_tracker import DeepByteTrack

class Tracker_metrics:
    def __init__(self, model_path, input_frames_path, output_tracking_path, batch_size=16):
        self.model = YOLO(model_path) 
        self.detections = None
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
        clear_output(wait=True)
        
    def deepbytetracker(self, reid_model_path,  
                        max_feature_age=30,
                        feature_distance_threshold=0.5,
                        track_activation_threshold=0.25,
                        lost_track_buffer=30,
                        minimum_matching_threshold=0.8,
                        frame_rate=30,
                        minimum_consecutive_frames=1,
                        num_classes=200):
        
        # Initialize DeepByteTracker
        tracker = DeepByteTrack(reid_model_path=reid_model_path,
                                max_feature_age=max_feature_age,
                                feature_distance_threshold=feature_distance_threshold,
                                num_classes=num_classes)
        
        # run tracking algorithm and save the output of tracking object in a txt file for evaluation with metrics
        tracker.deepbytetrack(detections=[self.detections],
                              output_tracking_path=self.ot_path, 
                              track_activation_threshold=track_activation_threshold,
                              lost_track_buffer=lost_track_buffer,
                              minimum_matching_threshold=minimum_matching_threshold,
                              frame_rate=frame_rate,
                              minimum_consecutive_frames=minimum_consecutive_frames)
        

    def bytetrack(self, 
                  track_activation_threshold=0.25,
                  lost_track_buffer=30,
                  minimum_matching_threshold=0.8,
                  frame_rate=30,
                  minimum_consecutive_frames=1):
        
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
                
                for frame_detection in detection_with_tracks:
                    bbox = frame_detection[0].tolist()
                    bbox = [round(value) for value in self.xyxy_to_xywh(bbox)]
                    line = f"{frame_num+1}, {frame_detection[4]}, {bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}\n"
                    f.write(line)
        print(f"ByteTrack tracking data saved to {output_path}")
    
    def deepsort(self,
                 max_age, 
                 n_init, 
                 max_cosine_distance, 
                 max_iou_distance,
                 embedder=None,
                 embedder_model_name=None,
                 embedder_wts=None):
        if embedder is None:
            deppsort_tracker = DeepSort(max_age=max_age,
                                        n_init=n_init,
                                        max_cosine_distance=max_cosine_distance,
                                        max_iou_distance=max_iou_distance)
        else:
            deppsort_tracker = DeepSort(max_age=max_age,
                                        n_init=n_init,
                                        max_cosine_distance=max_cosine_distance,
                                        max_iou_distance=max_iou_distance,
                                        embedder=embedder,
                                        embedder_model_name=embedder_model_name,
                                        embedder_wts=embedder_wts)
        frames = []
        total_frames = 0
        if os.path.isfile(self.if_path):
            cap = cv2.VideoCapture(self.if_path) 
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for _ in range(total_frames):
               ret, frame = cap.read()
               if ret:
                   frames.append(frame) 
        else:
            image_files = [f for f in os.listdir(self.if_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            image_files = natsort.natsorted(image_files)
            image_paths = [os.path.join(self.if_path, f) for f in image_files]
            total_frames = int(len(image_files))
            for image_path in image_paths:
                frames.append(cv2.imread(image_path))
        
        path = 'DS_tracker.txt'
        output_path = os.path.join(self.ot_path, path)
        with open(output_path, 'w') as f:                
            for frame_num in tqdm(range(total_frames), desc="DeepSort Processing"):
                detection = self.detections[frame_num]
                boxes_xyxy = detection.boxes.xyxy.cpu().numpy() 
                scores = detection.boxes.conf.cpu().numpy()
                class_ids = detection.boxes.cls.cpu().numpy()
                boxes_xywh = [self.xyxy_to_xywh(box) for box in boxes_xyxy]
                raw_detection = [(box, score, class_id) for box, score, class_id in zip(boxes_xywh, scores, class_ids)]
                
                # Pass detections to DeepSORT
                tracks = deppsort_tracker.update_tracks(raw_detections=raw_detection, frame=frames[frame_num])
                
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    bbox = track.to_ltrb()
                    bbox = [round(value) for value in self.xyxy_to_xywh(bbox)]
                    line = f"{frame_num+1}, {track.track_id}, {bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}\n"
                    f.write(line)
        print(f"DeepSort tracking data saved to {output_path}")
        
    def xyxy_to_xywh(self, boxes):
        x1 = boxes[0]
        y1 = boxes[1]
        x2 = boxes[2]
        y2 = boxes[3]
        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y1
        return [x, y, w, h]
    
    
    
    def single_object_tracker(self, tracker_type='CSRT', obj_cls='player', initial_bbox=None, display=False, is_rectangle=False):
        # Initialize the OpenCV tracker
        if tracker_type == 'CSRT':
            tracker = cv2.legacy.TrackerCSRT_create()
        elif tracker_type == 'KCF':
            tracker = cv2.legacy.TrackerKCF_create()
        elif tracker_type == 'MIL':
            tracker = cv2.legacy.TrackerMIL_create()
        else:
            raise ValueError(f"Unsupported tracker type: {tracker_type}")

        # Load frames
        frames = []
        if os.path.isfile(self.if_path):
            cap = cv2.VideoCapture(self.if_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for _ in range(total_frames):
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            cap.release()
        else:
            image_files = [f for f in os.listdir(self.if_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            image_files = natsort.natsorted(image_files)
            image_paths = [os.path.join(self.if_path, f) for f in image_files]
            for image_path in image_paths:
                frames.append(cv2.imread(image_path))

        if not frames:
            print("No frames found in the input path.")
            return
        
        # Define class mapping
        class_mapping = {
            "ball": 0,
            "player": 1,
            "referee": 2
        }
        
        # Initialize tracking
        if obj_cls in class_mapping:
            class_index = class_mapping[obj_cls]
            if initial_bbox is None:
                initial_detections = self.detections[0]
                for box in initial_detections.boxes:
                    if int(box.cls.item()) == class_index:
                        bbox = box.xyxy[0].cpu().numpy()
                        bbox = tuple(self.xyxy_to_xywh(bbox))
                        break
                else:
                    bbox = initial_detections.boxes.xyxy[0].cpu().numpy()
                    bbox = tuple(self.xyxy_to_xywh(bbox))
                    print(f"{obj_cls.capitalize()} not detected in the first frame. Taking the first detection.")
            else:
                bbox = initial_bbox
        else:
            print(f"Unsupported object class: {obj_cls}")
            return

        # Convert bbox to integer
        bbox = tuple(map(int, bbox))

        # Initialize tracker with the first frame and bounding box
        ok = tracker.init(frames[0], bbox)
        if not ok:
            print("Failed to initialize tracker.")
            return

        # Define the path to save tracking data
        path = f'SOT_tracker_{tracker_type}.txt'
        path = os.path.join('metrics', path)

        # Open a video writer if display is True
        output_path = os.path.join('output_videos', 'SOT_output.mp4') if display else None
        if display:
            height, width = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 25, (width, height))

        # Start tracking
        with open(path, 'w') as f:
            for idx, frame in enumerate(tqdm(frames, desc="Single Object Tracking")):
                ok, bbox = tracker.update(frame)

                if ok:
                    if bbox[0]>0 and bbox[1]>0 and bbox[2]>0 and bbox[3]>0 :
                        line = f"{idx+1}, {bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}\n"
                        f.write(line)
                        
                    if is_rectangle:
                        # Tracking success
                        p1 = (int(bbox[0]), int(bbox[1]))
                        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                    else:
                        cv2.ellipse(
                            frame,
                            center=(int(bbox[0] + (bbox[2] // 2)), (int(bbox[1] + bbox[3]))),
                            axes=(int(bbox[2]), int(0.35 * bbox[2])),
                            angle=0.0,
                            startAngle=-45,
                            endAngle=235,
                            color=(235, 206, 135),
                            thickness=2,
                            lineType=cv2.LINE_4
                        )
                else:
                    # Tracking failure
                    cv2.putText(frame, "Tracking failure detected", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

                # Save frame if display is True
                if display:
                    out.write(frame)

        if display:
            out.release()
            print(f"Single object tracking video saved to {output_path}")

        print("Single object tracking completed.")
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # def single_object_tracker(self, tracker_type='CSRT', obj_cls='player', initial_bbox=None, display=False, is_rectangle=False):
    #     # Initialize the OpenCV tracker
    #     if tracker_type == 'CSRT':
    #         tracker = cv2.legacy.TrackerCSRT_create()
    #     elif tracker_type == 'KCF':
    #         tracker = cv2.legacy.TrackerKCF_create()
    #     elif tracker_type == 'MIL':
    #         tracker = cv2.legacy.TrackerMIL_create()
    #     else:
    #         raise ValueError(f"Unsupported tracker type: {tracker_type}")

    #     # Load frames
    #     frames = []
    #     if os.path.isfile(self.if_path):
    #         cap = cv2.VideoCapture(self.if_path)
    #         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #         for _ in range(total_frames):
    #             ret, frame = cap.read()
    #             if ret:
    #                 frames.append(frame)
    #         cap.release()
    #     else:
    #         image_files = [f for f in os.listdir(self.if_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    #         image_files = natsort.natsorted(image_files)
    #         image_paths = [os.path.join(self.if_path, f) for f in image_files]
    #         for image_path in image_paths:
    #             frames.append(cv2.imread(image_path))

    #     if not frames:
    #         print("No frames found in the input path.")
    #         return
        
    #     # Define class mapping
    #     class_mapping = {
    #     "ball": 0,
    #     "player": 1,
    #     "referee": 2
    #     }
    #     # Initialize tracking
    #     if obj_cls == 'ball':
    #         class_index = class_mapping.get(obj_cls)
    #         if initial_bbox is None:
    #             initial_detections = self.detections[0]
    #             # check if ball is detected in the first frame
    #             for box in initial_detections.boxes:
    #                 if int(box.cls.item()) == class_index:
    #                     bbox = box.xyxy[0].cpu().numpy()
    #                     bbox = tuple(self.xyxy_to_xywh(bbox))
    #                     break
    #             # If ball is not detected in the first frame, take the first detection
    #             else: 
    #                  # For simplicity, take the first detection
    #                 bbox = initial_detections.boxes.xyxy[0].cpu().numpy()
    #                 bbox = tuple(self.xyxy_to_xywh(bbox))
    #                 print("Ball not detected in the first frame. Taking the first detection.")

    #         else:
    #             bbox = initial_bbox

    #     elif obj_cls == 'referee':
    #         class_index = class_mapping.get(obj_cls)
    #         if initial_bbox is None:
    #             initial_detections = self.detections[0]
    #             # check if referee is detected in the first frame
    #             for box in initial_detections.boxes:
    #                 if int(box.cls.item()) == class_index:
    #                     bbox = box.xyxy[0].cpu().numpy()
    #                     bbox = tuple(self.xyxy_to_xywh(bbox))
    #                     break
    #             # If referee is not detected in the first frame, take the first detection
    #             else: 
    #                 # For simplicity, take the first detection
    #                 bbox = initial_detections.boxes.xyxy[0].cpu().numpy()
    #                 bbox = tuple(self.xyxy_to_xywh(bbox))
    #                 print("Referee not detected in the first frame. Taking the first detection.")
    #         else:
    #             bbox = initial_bbox
                
    #     elif obj_cls == 'player':
    #         if initial_bbox is None:
    #             # Use the first frame's detections to initialize bbox
    #             initial_detections = self.detections[0]
    #             if len(initial_detections.boxes) == 0:
    #                 print("No detections in the first frame to initialize the tracker.")
    #                 return
    #             # For simplicity, take the first detection
    #             bbox = initial_detections.boxes.xyxy[0].cpu().numpy()
    #             bbox = tuple(self.xyxy_to_xywh(bbox))
    #         else:
    #             bbox = initial_bbox

    #     # Convert bbox to integer
    #     bbox = tuple(map(int, bbox))

    #     # Initialize tracker with the first frame and bounding box
    #     ok = tracker.init(frames[0], bbox)
    #     if not ok:
    #         print("Failed to initialize tracker.")
    #         return

        
    #         # Open a video writer if display is True
    #         if display:
    #             output_path = os.path.join(self.ot_path, 'single_object_tracking_output.mp4')
    #             height, width = frames[0].shape[:2]
    #             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #             out = cv2.VideoWriter(output_path, fourcc, 25, (width, height))

    #         # Start tracking
    #         for idx, frame in enumerate(tqdm(frames, desc="Single Object Tracking")):
    #             ok, bbox = tracker.update(frame)
    #             with open(path, 'w') as f:
    #                 line = f"{idx+1}, {bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}\n"
    #                 f.write(line)
    #                 if ok:
    #                     if is_rectangle:
    #                         # Tracking success
    #                         p1 = (int(bbox[0]), int(bbox[1]))
    #                         p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    #                         cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    #                     else:
    #                         cv2.ellipse(
    #                             frame,
    #                             center=(int(bbox[0]+(bbox[2]//2)),(int(bbox[1]+bbox[3]))),
    #                             axes=(int(bbox[2]), int(0.35*bbox[2])),
    #                             angle=0.0,
    #                             startAngle=-45,
    #                             endAngle=235,
    #                             color = (235,206,135),
    #                             thickness=2,
    #                             lineType=cv2.LINE_4
    #                         )
    #                 else:
    #                     # Tracking failure
    #                     cv2.putText(frame, "Tracking failure detected", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    #             print(f"{tracker_type} SOT tracking data saved to {path}")
    #             # Save frame if display is True
    #             if display:
    #                 out.write(frame)
            

    #         if display:
    #             out.release()
    #             print(f"Single object tracking video saved to {output_path}")

    #     print("Single object tracking completed.")    
    

    # def single_object_tracker(self, tracker_type='CSRT', initial_bbox=None, display=False):
    #     # Initialize the OpenCV tracker
    #     if tracker_type == 'CSRT':
    #         tracker = cv2.legacy.TrackerCSRT_create()
    #     elif tracker_type == 'KCF':
    #         tracker = cv2.legacy.TrackerKCF_create()
    #     elif tracker_type == 'MIL':
    #         tracker = cv2.legacy.TrackerMIL_create()
    #     else:
    #         raise ValueError(f"Unsupported tracker type: {tracker_type}")

    #     # Load frames
    #     frames = []
    #     if os.path.isfile(self.if_path):
    #         cap = cv2.VideoCapture(self.if_path)
    #         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #         for _ in range(total_frames):
    #             ret, frame = cap.read()
    #             if ret:
    #                 frames.append(frame)
    #         cap.release()
    #     else:
    #         image_files = [f for f in os.listdir(self.if_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    #         image_files = natsort.natsorted(image_files)
    #         image_paths = [os.path.join(self.if_path, f) for f in image_files]
    #         for image_path in image_paths:
    #             frames.append(cv2.imread(image_path))

    #     if not frames:
    #         print("No frames found in the input path.")
    #         return

    #     # Initialize tracking
    #     if initial_bbox is None:
    #         # Use the first frame's detections to initialize bbox
    #         initial_detections = self.detections[0]
    #         if len(initial_detections.boxes) == 0:
    #             print("No detections in the first frame to initialize the tracker.")
    #             return
    #         # For simplicity, take the first detection
    #         bbox = initial_detections.boxes.xyxy[0].cpu().numpy()
    #         bbox = tuple(self.xyxy_to_xywh(bbox))
    #     else:
    #         bbox = initial_bbox

    #     # Convert bbox to integer
    #     bbox = tuple(map(int, bbox))

    #     # Initialize tracker with the first frame and bounding box
    #     ok = tracker.init(frames[0], bbox)
    #     if not ok:
    #         print("Failed to initialize tracker.")
    #         return

    #     # Open a video writer if display is True
    #     if display:
    #         output_path = os.path.join(self.ot_path, 'single_object_tracking_output.mp4')
    #         height, width = frames[0].shape[:2]
    #         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #         out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

    #     # Start tracking
    #     for idx, frame in enumerate(tqdm(frames, desc="Single Object Tracking")):
    #         ok, bbox = tracker.update(frame)
    #         if ok:
    #             # Tracking success
    #             p1 = (int(bbox[0]), int(bbox[1]))
    #             p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    #             cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    #         else:
    #             # Tracking failure
    #             cv2.putText(frame, "Tracking failure detected", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    #         # Save frame if display is True
    #         if display:
    #             out.write(frame)

    #     if display:
    #         out.release()
    #         print(f"Single object tracking video saved to {output_path}")

    #     print("Single object tracking completed.")