# **Demo File: Object Detection and Tracking Report**

This demo file consists of three phases, each demonstrating a crucial aspect of computer vision-based tracking:

1. **Object Detection (Detection Phase)**
2. **Single Object Tracking (SOT Phase)**
3. **Multiple Object Tracking (MOT Phase)**

---

## **1. Object Detection (Detection Phase)**
### **Description:**
- In this phase, the system detects objects in video frames or images using a **pre-trained object detection model**.
- Each detected object is assigned a **bounding box**, along with a **class label and confidence score**.
- The detection phase serves as an input for the tracking stages.

### **Expected Outputs:**
- Bounding boxes around detected objects.
- Class labels and confidence scores displayed on the screen.
- Frames with visualized detection results.


### **Video Demonstration:**
Watch the video below to see the object detection phase in action, where bounding boxes are drawn around detected objects, and their class labels are displayed.

[**Object Detection Video**](/input_videos/v1.mp4)


<video width="640" height="480" controls>
  <source src="/input_videos/v1.mp4" type="video/mp4">.
</video>


---

## **2. Single Object Tracking (SOT Phase)**
### **Description:**
- After detection, a **single object** is selected and tracked across frames using a **Single Object Tracking (SOT) algorithm**.
- The selected object is continuously tracked using a bounding box, which updates dynamically.
- Algorithms like **CSRT, KCF, and Siamese Networks** are commonly used in this phase.

### **Expected Outputs:**
- A **single bounding box** around the selected object that updates across frames.
- Smooth tracking even when the object moves or slightly changes in appearance.


### **Video Demonstration:**
In this video, you can see how the algorithm tracks a single object in real time, updating the bounding box as the object moves through the frames.

[**Single Object Tracking Video in CSRT**](/output_videos/SOT_output_CSRT.mp4)

<video width="640" height="480" controls>
  <source src="/output_videos/SOT_output_CSRT.mp4" type="video/mp4">.
</video>

---

[**Single Object Tracking Video in MIL**](/output_videos/SOT_output_MIL.mp4)


<video width="640" height="480" controls>
  <source src="/output_videos/SOT_output_MIL.mp4" type="video/mp4">.
</video>

---

## **3. Multiple Object Tracking (MOT Phase)**
### **Description:**
- In this phase, **multiple detected objects** are tracked across frames using a **Multi-Object Tracking (MOT) algorithm**.
- Each tracked object is assigned a **unique ID**, which remains consistent throughout the video.

### **Expected Outputs:**
- Multiple bounding boxes tracking different objects.
- Unique ID assigned to each object for identification.
- Objects remain tracked even when they move or overlap.


### **Video Demonstration:**
The following video demonstrates how multiple objects are tracked, with each object receiving a unique ID and being continuously tracked even through occlusions and complex motion.

[**Multiple Object Tracking Video**](/output_videos/output_video.mp4)

<video width="640" height="480" controls>
  <source src="/output_videos/output_video.mp4" type="video/mp4">.
</video>


[**Multiple Object Tracking Video**](/output_videos/output_video_2.avi)

<video width="640" height="480" controls>
  <source src="/output_videos/output_video_2.avi" type="video/mp4">.
</video>


[**Multiple Object Tracking Video**](/output_videos/output_video_3.mp4)

<video width="640" height="480" controls>
  <source src="/output_videos/output_video_3.mp4" type="video/mp4">.
</video>
