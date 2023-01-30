# Camera Calibration

ROS package performing camera calibration employing a chessboard pattern and standard OpenCV routuines.


### saving image frames + camera poses

```
rosrun camera_calibration manual_save_camera_frames.py
```


### Intrinsics 

adjust argument inside the script depending on your system and chessboard pattern.

```
python calibrate_intrinsics.py
```


### Intrinsics + Extrinsics (eye_in_hand or eye_on_table)

```
python eye_in_hand_calib.py

python eye_on_table_calib.py
```


### Only Extrinsics

```
python eye_on_table_extrinsics_calib.py

```


### Only Extrinsics with from 3D camera


```
python eye_on_table_depth_extrinsics_calib.py

```
