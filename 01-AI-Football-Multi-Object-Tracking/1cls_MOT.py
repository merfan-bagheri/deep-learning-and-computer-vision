from utils import read_video, save_video
from trackers import Tracker_1cls
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner

import time
import sys


def main():
    
    folder_path = 'input_videos/f/'
    vid_name = 'f1'
    input_videos_path = f'{folder_path+vid_name}.mp4'
    anote_type = 'ellipse' # 'ellipse' for football and 'tri' for others
    video_frames = read_video(input_videos_path)
    


    # Initialize Tracker
    tracker = Tracker_1cls(model_path='models/f_v5s_1cls.pt', 
                    input_frames_path=input_videos_path,
                    output_tracking_path='metrics/',
                    batch_size = 2)

    tracker.detect_frames(confidence_level=0.4)
    
    tracks = tracker.bytetrack(track_activation_threshold=0.4,
                                lost_track_buffer=50,
                                minimum_matching_threshold=0.99,
                                frame_rate=25,
                                minimum_consecutive_frames=1)

    # Recompute ball positions after interpolation
    tracker.add_position_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frame = video_frames[0], player_detections = tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                    track['bbox'],
                                                    player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    output_video_frames = tracker.draw_annotations(video_frames, tracks, draw_type=anote_type)


    # Save video
    save_video(output_video_frames, f'output_videos/{vid_name}.mp4')
    

if __name__ == '__main__':
    main()