# Live_Face_Mask_Detection

Considering the current situation i have tried to build an amazing Computer vision project using Deep Neural Network ( self build CNN Architecture). In this project we can detect LIVE whether the person is wearing mask or not.<br>
And the amazing part is we can also connect to some external camera for detection by modifying few lines of code which i have comment down in my detection_app.py file.<br>

### STEPS
Initially i have imported my dataset and for better result data augmentation was done.<br>
In second step i have created my own CNN architecture to train the dataset (rather then importing Transfer learning models)<br>
After creating the model it was ready to use after writing few lines of code to read and predict the live images via webcam<br>

### Loss and Accuracy graph

![Screenshot (121)](https://user-images.githubusercontent.com/72625053/135704849-10a54a28-9404-441b-b5d6-62a364757b81.png)
![Screenshot (122)](https://user-images.githubusercontent.com/72625053/135704850-0c960e22-f3cf-4d72-890f-da5a1538e4df.png)

<br><b>Packages used</b> Tensorflow, Keras, Opencv, Matplotlib, Numpy etc.
<br><b>IDE</b> Pycharm

### Output prediction 

![Screenshot (120)](https://user-images.githubusercontent.com/72625053/135704900-767b4650-7feb-456b-b34a-f71d300eb5b2.png)

### Testing
![test](https://user-images.githubusercontent.com/72625053/135704691-0584b8e4-7bc7-4cf0-a97b-8f22671f593d.gif)
