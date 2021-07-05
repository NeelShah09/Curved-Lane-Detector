# Curved Lane Detection

This is a Computer Vision project created in python using OpenCV which calculates the curvature of road and offset of vehicle form centre of the lane.

## Steps for lane detection:

**1. Preprocessing :**
In preprocessing, colour filter is applied to create a mask of lane lines and then it is dilated using appropriate kernel.

**2. Perspective Transform :**
Perspective Transformation is applied on Region of interest to get an upright view of road for further processing.

**3. Lane Detection:**
First the start of the lane is detected at the bottom of the image and then sliding window algorithm is used to find the curved lane and a degree 2 polynomial is fit on the curve.

**4. Inverse Perspective Transform:**
The polygon form by 2 second degree polynomials is filled and projected back to normal perspective which represents the lane.

**5. Curvature and Offset Calculation:**
Final step is the calculation of radius of curvature and Offset of vehicle from centre. These calculations are carried out using basic calculus and some simple mathematics.

## Sample Output Video
<figure class="video_container">
    <iframe src="link" allowfullscreen="true" frameborder="0"></iframe>
</figure>
