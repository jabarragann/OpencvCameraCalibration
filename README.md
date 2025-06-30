# Camera Calibration with ChArUco Boards

## Install
```
pip install -e .
```

## Scripts

1. **video_to_frames.py**: Extracts frames from a video file.

```
python src/scripts/video_to_frames.py --input-video ./assets/sample_data/sample_video_7x10.mov --output-dir ./output/frames_5fps --target-fps 2 run
```
2. **calibrate.py**: Calibrates the camera using ChArUco boards
3. **undistort.py**: script to better understand undistortion in OpenCV 


## Links:

* [Brown conrady distortion model - shader toy](https://www.shadertoy.com/view/DslfR4)
* [Calibration with charuco board](https://medium.com/@nflorent7/a-comprehensive-guide-to-camera-calibration-using-charuco-boards-and-opencv-for-perspective-9a0fa71ada5f)