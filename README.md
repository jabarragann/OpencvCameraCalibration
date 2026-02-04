# Camera Calibration with Opencv (with regular and ChArUco Boards)

Simple scripts and documentation to calibrate cameras with Opencv.

Tested with Opencv 4.11.0.86 and Python 3.12.

## Install package
```
pip install -e .
```

## Calibration with regular chessboard

```bash
python sample_scripts/calibrate_single_camera_debug.py \
    --images_path ./assets/opencv_calib_tutorial/ \
    --camera left \
    --chessboard_rows 9 \
    --chessboard_cols 6 \
    --square_size 1.0 
```

## Calibration with ChArUco boards

1. **video_to_frames.py**: Extracts frames from a video file.
```
python src/cam_calib_lib/scripts/video_utils/video_to_frames.py \
    --input-video ./assets/sample_data_charuco/sample_video_7x10.mov \
    --output-dir ./output/frames_5fps \
    --target-fps 2 run 
```
2. **calibrate_camera.py**: Calibrates the camera using ChArUco boards
```
python src/cam_calib_lib/scripts/simple_demos/calibrate_camera.py
```
3. **undistort.py**: script to better understand undistortion in OpenCV 
```
python src/cam_calib_lib/scripts/simple_demos/undistort_images.py
```



## Links:

* [Brown conrady distortion model - shader toy](https://www.shadertoy.com/view/DslfR4)
* [Calibration with charuco board](https://medium.com/@nflorent7/a-comprehensive-guide-to-camera-calibration-using-charuco-boards-and-opencv-for-perspective-9a0fa71ada5f)