# Emotion Detection App
- The application «EMOTIONS DETECTION» realized is an Android  mobile application which detects and interprets a person's emotions,  and displays all the probabilities of each emotion, either from a photo  or from a video camera (real-time) using trained data.
## Emotion Recognition from Facial Expression 
- The project aims to train a model using tensorflow for facial emotion detection and used the trained model as predictor in android facial expression recongnition app.
- The project is Java-Machine Learning and it aims to classify the  emotion of a person's face into one of seven categories, using the  trained model as predictor, the model is trained using Tensorflow lite  and the dataset contain many 48x48 pixel faces expressing each  emotion from the seven ones: angry, disgusted, fearful, happy,  neutral, sad and surprised.


- The model is trained using  tensorflow python framework and used in android application where the basic langauge is java. 

- Basically tensorflow provides a c++ api, that can be used in android application. The trained model by python langauge can be integrated with android project  after inclduing tensorflow c++ framework dependencies and using native interface the model can be loaded and called in java class. This is the whole thing. 

## The total work of this project is divided into two parts :
1) Develop a <strong> Model </strong> in tensoflow from a <strong> DataSet </strong> using python langauge
  
2) Develop an android appication for facial expression recongtion 
  
### Part 1. Facial Expression Recongition Model developed in Tensorflow 

In this work , I have used a simple Convolutional Neural Network Architecture to train a facial expression dataset.

**1. DataSet:** The dataset is collected from Facial expression recognition challenge in kaggle
The challenge link https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/

The data consists of 48x48 pixel grayscale images of faces.<br>
The dataset contains facial expression of seven categories (Uncertain, Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised).

**2. Model:**
  
   We generated the <strong> Model </strong> from the <strong> DataSet </strong> using Python :
   <br>
   
    ``` from tflite_model_maker import image_classifier
    from tflite_model_maker.image_classifier import DataLoader

    # Load input data specific to an on-device ML app.
    data = DataLoader.from_folder('EmotionsFaces/')
    train_data, test_data = data.split(0.9)

    # Customize the TensorFlow model.
    model = image_classifier.create(train_data)

    # Evaluate the model.
    loss, accuracy = model.evaluate(test_data)

    # Export to Tensorflow Lite model and label file in `export_dir`.
    model.export(export_dir='/tmp/')
        ```
  
**3. Result:** 
The folder contains the generated <strong> model </strong> and <strong> Label </strong> : <br>
    https://github.com/Anas-Hilia/Emotion-Detection/tree/master/app/src/main/assets

### Part 2.  Facial Expression Recongition Application in Android

I have used Android Studio for this application. 

Integrating tensorflow dependency in android is really a tedious thing. the good news is that the latest news that android studio manages all dependencis related to tensorflow after adding the dependencies in *build.gradle(Module:app)* file 

```
dependencies {
    compile 'org.tensorflow:tensorflow-android:+' 
}

```

The final dependency part looks like 

```
dependencies {
    implementation fileTree(dir: 'libs', include: ['*.jar'])
    implementation 'com.android.support:appcompat-v7:26.1.0'
    implementation 'com.android.support.constraint:constraint-layout:1.1.0'
    testImplementation 'junit:junit:4.12'
    androidTestImplementation 'com.android.support.test:runner:1.0.1'
    androidTestImplementation 'com.android.support.test.espresso:espresso-core:3.0.1'
    compile 'org.tensorflow:tensorflow-android:+'
}
```

## Grafical Interface of The App : 
**1. App Logo**
  <div align="center">
  <img src="https://github.com/Anas-Hilia/Emotion-Detection/blob/master/screenshots/pic0.jpg?raw=true">
</div>
**2. Main Activity**
- The first layout which contain the image view where we  display the bitmap that we worked on and all buttons (icon format) that  assure the interaction between the user and the app : <br>
<div align="center">
  <img src="https://github.com/Anas-Hilia/Emotion-Detection/blob/master/screenshots/pic1.jpg?raw=true">
</div>

### When we click on picture icon button ,the app give us to choose between Taking photo, choose it from gallery or cancel:

<div align="center">
  <img src="https://github.com/Anas-Hilia/Emotion-Detection/blob/master/screenshots/pic2.jpg?raw=true">
</div>

### When we chose the picture we display it first ... Then :

1) If the picture contain face we classify it and we display the result  probabilities : <br>

Fearful | Happy | Neutral 
--- | --- | --- 
<img src="https://github.com/Anas-Hilia/Emotion-Detection/blob/master/screenshots/pic3.jpg?raw=true">|<img src="https://github.com/Anas-Hilia/Emotion-Detection/blob/master/screenshots/pic4.jpg?raw=true">|<img src="https://github.com/Anas-Hilia/Emotion-Detection/blob/master/screenshots/pic5.jpg?raw=true">
<br>

Sad | Angry | Surprised 
--- | --- | --- 
<img src="https://github.com/Anas-Hilia/Emotion-Detection/blob/master/screenshots/pic6.jpg?raw=true">|<img src="https://github.com/Anas-Hilia/Emotion-Detection/blob/master/screenshots/pic7.jpg?raw=true">|<img src="https://github.com/Anas-Hilia/Emotion-Detection/blob/master/screenshots/pic8.jpg?raw=true">
<br>


2) If not we display that message (No face detected in picture) : <br>
<div align="center">
  <img src="https://github.com/Anas-Hilia/Emotion-Detection/blob/master/screenshots/pic9.jpg?raw=true">
  <img src="https://github.com/Anas-Hilia/Emotion-Detection/blob/master/screenshots/pic10.jpg?raw=true">
</div>

### When we click on video icon button ,the app starts the camera view activity :

**3. CameraView Activity** 
- The second layout which contains the image view  where we display the bitmap got from camera continuously sequentially and we display the result too. <br>
<div align="center">
    <img src="https://github.com/Anas-Hilia/Emotion-Detection/blob/master/screenshots/pic11.jpg?raw=true">
</div>


