from utils import read_video, save_video
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner

import time
import sys


def main():
    """
    Main function to process a video for object tracking, camera movement estimation, 
    view transformation, speed and distance estimation, team assignment, and ball acquisition.
    Steps:
    1. Read the input video.
    2. Initialize the object tracker and get object tracks.
    3. Add positions to the tracks.
    4. Estimate camera movement and adjust object positions accordingly.
    5. Transform the view and add transformed positions to the tracks.
    6. Interpolate ball positions.
    7. Estimate speed and distance for the tracked objects.
    8. Assign teams to players based on their appearance.
    9. Assign ball possession to players.
    10. Draw annotations, camera movement, and speed/distance on the video frames.
    11. Save the processed video to the output file.
    """

    folder_path = 'input_videos/f/'
    vid_name = 'f1'
    input_videos_path = f'{folder_path+vid_name}.mp4'
    
    video_frames = read_video(input_videos_path)
    


    # Initialize Tracker
    tracker = Tracker(model_path='models/f_v5s_3cls.pt', 
                    trajectory_length=20, 
                    draw_trajectory=True,
                    input_frames_path=input_videos_path,
                    output_tracking_path='metrics/',
                    batch_size = 8)

    tracker.detect_frames(confidence_level=0.5)
    
    tracks = tracker.bytetrack(track_activation_threshold=0.5,
                                lost_track_buffer=100,
                                minimum_matching_threshold=0.99,
                                frame_rate=25,
                                minimum_consecutive_frames=1)

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    # Recompute ball positions after interpolation
    tracker.add_position_to_tracks(tracks)

    # Assign Player Teams
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frame = video_frames[0], 
                                    player_detections = tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],   
                                                    track['bbox'],
                                                    player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]



    # Assign Ball Aquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control= []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True


    output_video_frames = tracker.draw_annotations(video_frames, tracks)


    # Save video
    save_video(output_video_frames, f'output_videos/{vid_name}.mp4')
    

if __name__ == '__main__':
    main()