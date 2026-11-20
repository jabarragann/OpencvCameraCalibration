# Microscope stereo calibration notes

## Context

Goal: calibrate the Haag-Streit stereo microscope (high magnification, baseline ≈ 26 mm) so its
rectified image pairs can be fed to **FoundationStereo** for disparity / depth estimation.

Dataset: `datasets/microscope_dataset7/C2` (ChArUco 10×7, 5 mm squares, 3 mm markers),
46 usable stereo pairs after `--exclude "0008 0027 0033 0034 0048 0051"`.

Calibration result (2026-11-17), with `--distortion_flags CALIB_FIX_K2 CALIB_FIX_K3 CALIB_ZERO_TANGENT_DIST`
and `--stereo_flags CALIB_USE_INTRINSIC_GUESS`:

- Stereo reprojection error (RMS): 0.86 px
- Baseline $|T| = 26.6$ mm, $T \approx (-26.4,\ 1.1,\ 2.4)$ mm
- $f_x, f_y \approx 8990$ px for both cameras, $k_1 \approx -0.8$ to $-0.9$
- Rotation between the cameras: ~3° toe-in, ~2.4° roll
- Triangulated board distance: 426–487 mm (median 463 mm)

## The two problems with the default rectification

The calibration itself was fine. The trouble came from **rectification**: `cv2.stereoRectify` was
called with `flags=cv2.CALIB_ZERO_DISPARITY` and `alpha=0`. That caused two problems:

### Problem 1: huge disparities

After rectification the board was **795–910 px** apart between left and right images.
FoundationStereo's default maximum disparity is 416 px (from memory of its config; double-check),
so these disparities can't be estimated at all.

### Problem 2: heavy crop

The rectified focal length jumped from ≈ 8990 px to **14575 px**, i.e. a **1.62× zoom**.
About 38% of the field of view was lost in each direction, and pixel errors measured in the
rectified images were inflated by 1.62× (1.35 px rectified ≈ 0.83 px native).

### Root cause: converging optical paths

The two optical paths of the microscope **converge** (toe-in, crossing ≈ 0.5 m away). In the raw
images the board is only **13–60 px** apart between left and right.

`CALIB_ZERO_DISPARITY` is meant for **parallel** camera rigs. It forces both rectified images to
share the same principal point, so rectification has to undo the convergence completely:

- the full parallel-camera disparity $f B / Z$ appears in the images → huge disparities;
- the two views are shifted so far apart sideways that they only share a narrow strip, and
  `alpha=0` zooms in to keep only valid pixels → heavy crop.

## What `CALIB_ZERO_DISPARITY` does

After rectification both images behave like two parallel cameras looking straight ahead. The only
remaining choice is where each rectified image's principal point $c_x$ sits.

**With `CALIB_ZERO_DISPARITY`**, both images share the same principal point
($c_{x,L} = c_{x,R}$). A point at infinity has disparity 0, and

$$
d = x_L - x_R = \frac{f\,B}{Z}
\qquad\Longrightarrow\qquad
Z = \frac{f\,B}{d}
$$

**With `flags=0`**, OpenCV may give each image its own $c_x$ to keep as much of both views as
possible. The images are shifted sideways relative to each other, so

$$
d = \frac{f\,B}{Z} - d_{\text{offs}},
\qquad
d_{\text{offs}} = c_{x,R} - c_{x,L}
$$

$$
Z = \frac{f\,B}{d + d_{\text{offs}}}
$$

The depth information is identical in both cases; it's just split differently between the
measured disparity $d$ and the constant offset $d_{\text{offs}}$.

**Catch:** `CALIB_ZERO_DISPARITY` is the *default* value of `flags` in `cv2.stereoRectify`.
Deleting the argument changes nothing; you must pass `flags=0` explicitly.

## Comparison of rectification options (C2 data)

Measured by running the ChArUco corners of all 46 pairs through each rectification option.

| Rectification | Rectified $f$ (zoom) | Valid area | Board disparity min / median / max |
|---|---|---|---|
| `ZERO_DISPARITY`, alpha=0 (old default) | 14575 (1.62× crop) | full frame | **795 / 836 / 910 px** |
| `ZERO_DISPARITY`, alpha=1 | 6430 (downsampled) | small center area, black borders | 351 / 369 / 401 px |
| **`flags=0`, alpha=0 (recommended)** | **9523 (1.06×)** | **full frame, no black borders** | **26 / 53 / 101 px** |
| `flags=0`, alpha=1 | 8138 | black borders | −49 / −27 / 15 px (negative) |

With `flags=0, alpha=0`, the rectified principal points are $c_{x,L} = 1018$ and $c_{x,R} = 1511$,
so $d_{\text{offs}} \approx 493$ px. Full calibration rerun with this setting: mean rectified
$|dy|$ = 0.94 px over 2161 corners (vs 1.35 px over ~870 corners with `ZERO_DISPARITY`).

## Recommendation for FoundationStereo

Use **`flags=0`, `alpha=0`**:

- **Disparity range:** separate principal points absorb the sideways shift, so disparities fall in a
  small positive range (26–101 px for the board).
- **Image quality:** almost no crop (1.06×), the full frame is valid, no black borders for the
  network to choke on.
- **Why not a higher alpha:** with `flags=0`, higher alpha only adds black borders and pushes
  disparities negative. FoundationStereo can't output negative disparities.

In this repo: pass `--rectify_flags` with no values to `calibrate_stereo_camera_charuco`
(default is `CALIB_ZERO_DISPARITY`). The GUI in
`src/cam_calib_lib/scripts/stereo_tools/rectify_image_gui.py` still hardcodes
`CALIB_ZERO_DISPARITY`.

### Converting FoundationStereo disparity to depth

$$
Z = \frac{f\,B}{d + d_{\text{offs}}},
\qquad
f = 9523\ \text{px},\quad B = 26.6\ \text{mm},\quad d_{\text{offs}} \approx 493\ \text{px}
$$

The values come from the rectified projection matrices in `left.yaml` (`P1`) and `right.yaml` (`P2`):

$$
f = P_1[0,0],
\qquad
B = -\frac{P_2[0,3]}{P_2[0,0]},
\qquad
d_{\text{offs}} = P_2[0,2] - P_1[0,2]
$$

Alternatively, `cv2.reprojectImageTo3D` with the `Q` matrix from `stereoRectify` already includes
the offset. Note: `stereo_calibration_pipeline` currently discards `Q`, so it isn't saved yet.

### Things to watch

- **Scene beyond ≈ 512 mm:** disparity 0 corresponds to
  $Z_0 = f B / d_{\text{offs}} \approx 512$ mm. Anything farther would need a negative disparity.
  The board was at 426–487 mm and the microscope's shallow depth of field probably keeps the scene
  in that band. If you refocus farther or have distant background, increase $c_{x,R}$ slightly to
  leave margin, then recompute $d_{\text{offs}}$ and `P2` / `Q` to match.
- **Resizing for FoundationStereo:** if the 1920×1080 images are scaled by $s$ before inference,
  scale $d$, $f$ and $d_{\text{offs}}$ by the same factor $s$ ($B$ is unchanged).
- **Sanity-check the distance:** compare the triangulated board distance (≈ 463 mm) with the
  microscope's actual working distance. A matching baseline does not validate $f$: with a narrow
  field of view, $f$ and $Z$ can scale together while $B$ stays the same.
