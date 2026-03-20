# Calibration with OpenCV 

## Distortion models
2 params --> No tangent and K3 fixed
4 params --> K3 fixed
5 params --> tangent(p1,p2) and K1,K2 and K3 active (Default in Opencv. Flags=0)

## Notes:
* Sometimes the 2 parameter setup (defined by the flags below) will prevent the stereo calibration to
work.

```bash
--distortion_flags CALIB_FIX_K3 CALIB_ZERO_TANGENT_DIST \
--stereo_flags CALIB_FIX_INTRINSIC
```


To fix it use the `CALIB_USE_INTRINSIC_GUESS` in the stereo flags.
