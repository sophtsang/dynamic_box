## Installation:
### Conda: 
https://docs.anaconda.com/anaconda/install/linux/:
Download in Ubuntu home directory:
``` 
Anaconda3-2024.06-1-Linux-x86_64.sh from https://repo.anaconda.com/archive/
```

Run:
```
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.06-1-Linux-x86_64.sh
bash ~/<Wherever you downloaded it>/Anaconda3-2024.06-1-Linux-x86_64.sh
```

Anaconda is now downloaded in ``` /home/<USER>/anaconda3 ``` in Ubuntu

Refresh terminal: ``` source ~/.bashrc ```

### Activate conda environment:
``` python
conda create -n env_stereo python=3.10
conda activate env_stereo
conda install pytorch
conda install cuda80 -c pytorch
conda install torchvision -c pytorch
```


## How This Works:
ZED camera captures stereo pairs of left (Il) and right (Ir) images -> run through a 2D detector that generates 2D boxes on regions of interest (ROIs) in Il and Ir:
1) 2D detector that generates 2D boxes in Il and Ir:
   Given stereo image pair input: an Il, Ir pair from ZED -> identify left (l) and right (r) ROIs -> threshold -> aquire m, n ROIs in Il, Ir ->
   perform association with SSIM(l,r) for each ROI pair combination 
   (assume l, r are similar).
2) Box association algorithm matches obejct detections across both images,
    forms a box association pair if SSIM threshold satisfied.
3) Stereo regression: with left-right box association pairs, apply ROI
   Align -> concatenate l, r into two fully-connected layers + ReLU
   layer. 
4) Given left-right 2D boxes, perspective keypoint, regressed dim, generate
   3D box: look at ```3D box Estimation``` for reference on how we will use the 
   left and right 2D boxes to generate a 3D box. 

### Steps completed:
1) Implemented YOLOv3 and YOLOv11 2D bounding box generators for objects in stereo images. Returns bounding box information in the form of $(x_l,y_l,w,h)$.

### Currently working on:
1) Figure out 8-point algorithm or use given ZED intrisinc, extrinsic parameters to derive Fundamental matrix F.
2) Use F, ZED camera instrinsic matrices $K_R$ and $K_L$ to derive essential matrix $E={K_R^T}\cdot{F}\cdot{K_L}$
3) __FIGURE OUT THE MATH FOR THIS:__ Rectify the stereo images with epipolar geometry and epilines so that corresponding points between the two images lie on the same horizontal line (epipolar line).
4) Derive depth of objects (most likely the corners of each 2D bounding box) in the rectified stereo images. Use the formula $Z=\dfrac{{f}\cdot{B}}{d}$, where $Z$ is distance of the object from the camera, $f$ is focal length, $B$ is baseline, and $d$ is disparity (horizontal shift between left and right image). Given a point $P$ that corresponds in the left and right images, disparity $d=x_l-x_r$, where $x_l$ is the horizontal pixel distance of $P$ in the left image, and $x_r$ is the horizontal pixel distance of $P$ in the right image.

   _Only need to calculate horizontal disparity between stereo images have been rectified, so we ignore vertical disparity._
6) Estimate 3D coordinates with depth and ZED camera calibration information. The 3D position $(X,Y,Z)$ of the front face of the object is calculated by: $X={x-c_x}\cdot{\dfrac{Z}{f_x}}$, $Y={y-c_y}\cdot{\dfrac{Z}{f_y}}$, and $Z=depth(x,y)$ where $(x,y)$ is the pixel coordinates of the corners of the 2D bounding box, focal lengths of the left ZED camera $f_x=700.819$ and $f_y=700.819$, and principle point or optical center coordinates of the left ZED camera $c_x=665.465$ and $c_y=371.953$. The back face of the object is estimated with the object's depth and width.

## Eventual ROS2 Package Implementation:
...

## Important links:
[ZED Calibration File](https://support.stereolabs.com/hc/en-us/articles/360007497173-What-is-the-calibration-file)

[CS231A Course Notes 3: Epipolar Geometry](https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf)

[Complex YOLO: YOLOv4 for 3D Object Detection](https://medium.com/@mohit_gaikwad/complex-yolo-yolov4-for-3d-object-detection-3c9746281cd2)

[Stereo R-CNN based 3D Object Detection](https://github.com/HKUST-Aerial-Robotics/Stereo-RCNN?tab=readme-ov-file)

[3D Reconstruction and Epipolar Geometry](https://github.com/laavanyebahl/3D-Reconstruction-and-Epipolar-Geometry)

[The 8-point algorithm](https://www.cs.cmu.edu/~16385/s17/Slides/12.4_8Point_Algorithm.pdf)
