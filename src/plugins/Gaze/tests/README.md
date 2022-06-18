# Gaze testing

We tested the fotofaces algorithm and another algorithm from the pysource website

## Old fotofaces


### fotofaces algorithm Results

1  - > 92.35
2  - > 95.75
3  - > 55.44
4  - > 58.05
5  - > 84.01   
6  - > 96.27   ( wrong)
7  - > 82.92
8  - > 81.55
9  - > 82.82
10 - > 95.87  

focus > 90


![test 1](https://github.com/FotoFaces/Fotofaces/src/plugins/Gaze/tests/images/test_fotofaces.png)

This algorithm doesn't pass a person looking up, right or left, but it fails when a person is looking down, possible because the pupil is in the center of the eye despite the person is looking down



## pysource algorithm


### pysource algorithm Results

1  - > 0.827
2  - > 0.928
3  - > 0.534
4  - > 2.686
5  - > 0.964   (wrong)
6  - > 1.070   (wrong)
7  - > 0.630
8  - > 1.600
9  - > 0.597
10 - > 0.878  


0.80 < gaze_ratio < 1.550

![test 2](https://github.com/FotoFaces/Fotofaces/src/plugins/Gaze/tests/images/test_pysource.png)

This algorithm fails when the person is looking above and down, it detects if a person is looking forward and not close to the camera

<hr>

<h3>Conclusion</h3> 

O melhor algoritmo é o fotofaces apesar de aceitar pessoas a olharem para baixo



