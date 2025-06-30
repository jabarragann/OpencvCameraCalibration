## Thoughts about Opencv undistortion notes

Why does the opencv undistortion function sends some pixels in the center to the border of the image?

This results are generated with a 3 coefficient radial distortion model and 2 coefficient tangential distortion model.  The problem is coming from the K_3 coefficient which goes with the 6th order polynomial. Fixing this to 0 helps.  Notice how the free parameter could hide the error in the undistortion process.

Big changes in the resolution of the image after undistortion, could be an indication that the estimated distortion model is not appropriate for the camera.

Free parameter 0
![question](images/free_param0.png)

Free parameter 1
![question](images/free_param1.png)
