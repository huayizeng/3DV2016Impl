## Camera Matrix for test.png
### K
Matrix(((1050.0, 0.0, 480.0),
        (0.0, 1050.0, 270.0),
        (0.0, 0.0, 1.0)))
### RT
Matrix(((0.6859206557273865, 0.7276763319969177, -4.011331711240018e-09, -0.3960070312023163),
        (0.32401347160339355, -0.3054208755493164, -0.8953956365585327, 0.3731381893157959),
        (-0.6515582203865051, 0.6141703724861145, -0.44527140259742737, 11.250574111938477)))

## Steps towards implementations:
- [] Annotate 2D coordinates for test.png. Store every single line as x1, y1, x2, y2
- [] Read 3dv2016.obj and project onto 2D plane, to assure the projection matrices are correct. (TODO: Huayi)
- [] Set world coordinates

## For the first task, the defined K and RT are:
K = np.asarray([[450.0, 0.0, 480.0],
        [0.0, 450.0, 270.0],
        [0.0, 0.0, 1.0]])

RT = np.asarray([[ -8.99912260e-01, 5.27725762e-02, 9.61641215e-01, 75],
                 [ -2.12500000e-01, 7.22500000e-01, -1.11250000e-01, 0],
                 [ -2.98394327e-01,  -6.54094837e-01, -6.04869715e-02, -200]])
