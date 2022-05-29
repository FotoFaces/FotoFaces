# Test Head Pose

## Testing Old fotofaces algorithm

### Normal Results

![normal test result](https://github.com/FotoFaces/Fotofaces/src/plugins/HeadPose/tests/images/test_20.png)

With the old fotofaces classifying if a person is facing forward by the angle being less than 20 is a little naive because when the face is turning left the angle becomes negative 

### Results with absolute value

![abs test result](https://github.com/FotoFaces/Fotofaces/src/plugins/HeadPose/tests/images/test_abs_20.png)

When returning the absolute value of the angle of the person the results become a bit better, however it still fails on one test due not being strict enough 

### Results with Reduce tolerance range

![15 abs test result](https://github.com/FotoFaces/Fotofaces/src/plugins/HeadPose/tests/images/test_abs_15.png)

By reducing the tolerance range the algorithm pass all the tests


## Observation

![15 abs test result](https://github.com/FotoFaces/Fotofaces/src/plugins/HeadPose/tests/images/head_pose1.jpg)

This image produced this score:
pitch 2.4826839779416647
roll 2.32299356298174
yaw 3.238287507312369

So, if we want a very strict algorithm (make real sure that the person is facing the camera) i suject a tolerance range of 5 (limit)