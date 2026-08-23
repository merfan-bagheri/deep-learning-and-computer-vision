import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import supervision as sv


# Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Output size: (B, C, 1, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Output size: (B, C, 1, 1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze: Channel-wise global spatial average and max pooling
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        # Excitation: Sum and apply sigmoid activation
        out = avg_out + max_out
        out = self.sigmoid(out)
        # Scale: Multiply input by channel attention map
        return x * out

# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Ensure kernel size is odd
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze: Channel-wise AvgPool and MaxPool along channel axis
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Shape: (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Shape: (B, 1, H, W)
        # Concatenate average and max pool outputs
        x_cat = torch.cat([avg_out, max_out], dim=1)  # Shape: (B, 2, H, W)
        # Convolutional layer
        x_conv = self.conv(x_cat)
        # Scale: Apply sigmoid activation
        x_sa = self.sigmoid(x_conv)
        # Multiply input by spatial attention map
        return x * x_sa

# Residual Block with CBAM (Channel and Spatial Attention)
class ResidualCBAMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualCBAMBlock, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # Attention modules
        self.ca = ChannelAttention(out_channels)
        self.sa = SpatialAttention()
        # Shortcut connection
        self.downsample = downsample

    def forward(self, x):
        identity = x

        # First convolutional layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second convolutional layer
        out = self.conv2(out)
        out = self.bn2(out)

        # Apply Channel Attention
        out = self.ca(out)
        # Apply Spatial Attention
        out = self.sa(out)

        # Add residual (shortcut) connection
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        out = self.relu(out)
        return out

# Main Network with Residual CBAM Blocks
class CustomCBAMNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomCBAMNet, self).__init__()
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # Max pooling
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Residual CBAM layers
        self.layer1 = self._make_layer(64, 64, blocks=2)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        # Adaptive average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # self.fc1 = nn.Linear(512, num_classes)
        self.fc = nn.Linear(512, num_classes)
        # self.fc2 = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        # Handles downsampling if dimensions change
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        # First block may need downsampling
        layers.append(ResidualCBAMBlock(in_channels, out_channels, stride, downsample))
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(ResidualCBAMBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual CBAM layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Classification head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class DeepByteTrack:
    def __init__(self, reid_model_path,  
                 max_feature_age=30,
                 feature_distance_threshold=0.5,
                 num_classes=100):
        
        self.max_feature_age = max_feature_age
        self.feature_distance_threshold = feature_distance_threshold  # Threshold for feature matching
        
        # Initialize ReID feature storage
        # Map of track ID to {'features': tensor, 'age': int}
        self.id_features = {}
        
        # Mapping from ByteTrack IDs to consistent IDs
        self.track_id_mapping = {}
        self.next_id = 0  # Next available unique ID
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reid_model = CustomCBAMNet(num_classes) 
        self.reid_model.load_state_dict(torch.load(reid_model_path))
        self.reid_model.to(self.device)
        self.reid_model.eval()
        
        # Define image transformations for ReID model input
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # Adjust size as per your model's requirement
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Use appropriate mean and std for your dataset
                                 std=[0.229, 0.224, 0.225])
        ])
        
    def deepbytetrack(self, detections, 
                      output_tracking_path, 
                      track_activation_threshold=0.25,
                      lost_track_buffer=30,
                      minimum_matching_threshold=0.8,
                      frame_rate=30,
                      minimum_consecutive_frames=1):
        path = 'DBT_tracker.txt'
        output_path = os.path.join(output_tracking_path, path)
        bytetrack_tracker = sv.ByteTrack(track_activation_threshold=track_activation_threshold,
                                         lost_track_buffer=lost_track_buffer,
                                         minimum_matching_threshold=minimum_matching_threshold,
                                         minimum_consecutive_frames=minimum_consecutive_frames,
                                         frame_rate=frame_rate)
        
        with open(output_path, 'w') as f:
            for frame_num, detection in tqdm(enumerate(detections[0]), total=len(detections[0]), desc="DeepByteTrack Processing"):
                # Convert to supervision Detection format
                detection_supervision = sv.Detections.from_ultralytics(detection)
                
                # Track objects using ByteTrack
                detections_with_tracks = bytetrack_tracker.update_with_detections(detection_supervision)
                
                # List to keep track of current IDs in this frame
                current_ids = []
                
                # Process each detection with track ID
                for det in detections_with_tracks:
                    bbox = det[0].tolist()
                    byte_track_id = int(det[4].item())
                    bbox_xyxy = [int(round(value)) for value in bbox]
                    
                    # Crop image for ReID feature extraction
                    orig_img = detection.orig_img  # NumPy array in 0-255, shape HxWx3
                    crop_img = self.crop_image(orig_img, bbox_xyxy)
                    
                    # Extract feature using ReID network
                    feature = self.extract_feature(crop_img)
                    # print(self.track_id_mapping)
                    # Check if ByteTrack ID is already mapped to our consistent ID
                    if byte_track_id in self.track_id_mapping:
                        track_id = self.track_id_mapping[byte_track_id]
                        self.update_id_features(track_id, feature)
                    else:
                        # New ByteTrack ID, attempt to match with existing IDs
                        matched_id = self.match_feature(feature)
                        if matched_id is not None:
                            # Reassign ByteTrack ID to matched consistent ID
                            self.track_id_mapping[byte_track_id] = matched_id
                            track_id = matched_id
                            self.update_id_features(track_id, feature)
                        else:
                            # Assign a new consistent ID
                            track_id = self.next_id
                            self.next_id += 1
                            self.track_id_mapping[byte_track_id] = track_id
                            self.id_features[track_id] = {'features': feature, 'age': 0}
                    
                    current_ids.append(track_id)
                    
                    # Save the tracking result
                    bbox_xywh = self.xyxy_to_xywh(bbox_xyxy)
                    line = f"{frame_num+1}, {track_id}, {bbox_xywh[0]}, {bbox_xywh[1]}, {bbox_xywh[2]}, {bbox_xywh[3]}\n"
                    f.write(line)
                
                # Update the age of IDs and remove old IDs
                self.update_id_ages(current_ids)
                
            print(f"DeepByteTrack tracking data saved to {output_path}")
                
    def extract_feature(self, image):
        # Convert the cropped image to a tensor
        img_tensor = self.transform(image)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
        
        with torch.no_grad():
            feature = self.reid_model(img_tensor)
        
        # Normalize the feature vector
        feature = feature / feature.norm(p=2, dim=1, keepdim=True)
        
        return feature.cpu()
    
    def update_id_features(self, track_id, new_feature):
        if track_id in self.id_features:
            # Update features using a moving average
            old_feature = self.id_features[track_id]['features']
            updated_feature = 0.7 * old_feature + 0.3 * new_feature  # Adjust weights as needed
            updated_feature = updated_feature / updated_feature.norm(p=2, dim=1, keepdim=True)  # Re-normalize
            self.id_features[track_id]['features'] = updated_feature
            self.id_features[track_id]['age'] = 0  # Reset age
        else:
            # New ID, add to feature storage
            self.id_features[track_id] = {'features': new_feature, 'age': 0}
            
    def update_id_ages(self, current_ids):
        # Increment age of all IDs not in current frame
        ids_to_remove = []
        for id_ in list(self.id_features.keys()):
            if id_ not in current_ids:
                self.id_features[id_]['age'] += 1
                if self.id_features[id_]['age'] > self.max_feature_age:
                    ids_to_remove.append(id_)
        # Remove IDs that have exceeded max age
        for id_ in ids_to_remove:
            del self.id_features[id_]
            # Remove from track ID mapping
            byte_track_ids_to_remove = [k for k, v in self.track_id_mapping.items() if v == id_]
            for k in byte_track_ids_to_remove:
                del self.track_id_mapping[k]
            
    def match_feature(self, new_feature):
        # Compare new feature to existing features to find a match
        min_distance = float('inf')
        matched_id = None
        for id_, data in self.id_features.items():
            stored_feature = data['features']
            distance = self.compute_distance(new_feature, stored_feature)
            if distance < self.feature_distance_threshold and distance < min_distance:
                min_distance = distance
                # print(id_)
                matched_id = id_
        return matched_id
    
    def compute_distance(self, feature1, feature2):
        # Calculate cosine distance between two features
        similarity = torch.nn.functional.cosine_similarity(feature1, feature2)
        distance = 1 - similarity.item()
        return distance
    
    def crop_image(self, image, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy
        crop_img = image[y1:y2, x1:x2]  # Assuming image is NumPy array HxWx3
        return crop_img
    
    def xyxy_to_xywh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy
        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y1
        return [x, y, w, h]
