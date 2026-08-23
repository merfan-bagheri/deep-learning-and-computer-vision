def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int((y1+y2)/2)

def get_bbox_width(bbox):
    return bbox[2]-bbox[0]

def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def measure_xy_distance(p1,p2):
    return p1[0]-p2[0],p1[1]-p2[1]

def get_foot_position(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int(y2)

def xyxy_to_xywh(boxes):
    x1 = boxes[0]
    y1 = boxes[1]
    x2 = boxes[2]
    y2 = boxes[3]
    x = x1
    y = y1
    w = x2 - x1
    h = y2 - y1
    return [x, y, w, h]

def process_bounding_boxes(input_file, output_file, track_id):
    bounding_boxes = []
    first_frame_box = None

    with open(input_file, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            frame_number = int(parts[0])
            track_id_current = int(parts[1])

            # Check if the current track_id matches the specified one
            if track_id_current == track_id:
                # Keep only frame number and bounding box coordinates
                bounding_box = parts[2:6]  # Extract x, y, width, height
                bounding_boxes.append(f"{frame_number}, {', '.join(bounding_box)}")
                
                # Store the bounding box for the first frame
                if frame_number == 1:
                    first_frame_box = bounding_box

    # Save only the lines for the specified track_id to the new file
    with open(output_file, 'w') as file:
        for box in bounding_boxes:
            file.write(box + '\n')

    # Return the bounding box for the specified track_id from the first frame
    return first_frame_box