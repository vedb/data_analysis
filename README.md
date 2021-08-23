# Data Analysis
## A data analysis and visualization repository for the visual experience data base
This repository has four major components:
* **Gaze**
  * Pupil and gaze tracking models and scripts are included here. 
  * From the well-known methods we have either implemented or under implementation:
    * [pupil-labs](https://github.com/pupil-labs/pupil)
    * [RITnet](https://github.com/KamranBinaee/RGnet/tree/master/rgnet)
    * [DeepVog](https://github.com/pydsgz/DeepVOG)
    * [EllsegI and II](https://github.com/RSKothari/EllSeg) To be added!

* **Odometry**
  * The pipeline in order to bring T265 head pose tracking data into bilogically correct frame of reference. 

* **Scene**
  * Marker detection pipeline
    * Pupil-lab's circular marker
    * 8x9 checkerboard
    * April-tag markers
  * Mediapipe model for:
    * Hand tracking
    * Face tracking
    * Face mesh tracking
    * Body pose tracking
* **visualization**
  * Pupil tracking visualization:
    * Pupil-labs pipeline: The ellipse and confidence values
    * RITnet pipeline: pupil, iris, scelera, skin masks and ellipse fit confidence values
    * DeepVog pipeline: pupil region, 3D gaze vector, network output mask
  * Gaze overlaid video:
    * World and super imposed eye videos plus the gaze point
    * World and the detected marker positions
  * Detected object bounding box (Yolo)
  * Detected object by mediapipe
  * Eye image annotation tool:
    * This tool is used to create semantic segmentation mask for eye images as ground truth 
## Development milestones and projects:
* **[Data Loading](https://github.com/vedb/data_analysis/milestone/1)**
  * Improve performance during loading
* **[Add April Tag detection Pipeline](https://github.com/vedb/data_analysis/milestone/7)**
  * Merge from local branch
* **[Add Intrinsics and Extrinsics Pipeline](https://github.com/vedb/data_analysis/milestone/6)**
  * Merge from local branch and refactor as one script
* **[Add 3D Gaze Calibration Pipeline](https://github.com/vedb/data_analysis/milestone/5)**
  * Incorporate Pupil-lab's latest pye3d into the repo
* **[DeepVog Pupil Tracking Pipeline](https://github.com/vedb/data_analysis/milestone/4)**
* **[RITnet Pupil Tracking Pipeline](https://github.com/vedb/data_analysis/milestone/3)**
  * Merge from local [repo](https://github.com/KamranBinaee/RGnet/tree/master/rgnet)
* **[Data Saving](https://github.com/vedb/data_analysis/milestone/2)**
  * Run-time performance improvement:
    * Test and use other formats i.e. pandas, h5py, [etc](https://github.com/RSKothari/Data2H5).   



