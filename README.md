# custom_face_detection_model
Developing a custom face detection model from scratch using custom data and the Tensorflow Keras Functional API.

## 1. Get Data

### 1.1 Video Capturing
In this stage, I captured images of myself using OpenCV.

```python
IMAGES_PATH = os.path.join('data','images')
number_images = 30

cap = cv2.VideoCapture(0)
for imgnum in range(number_images):
    print('Collecting image {}'.format(imgnum))
    ret, frame = cap.read()
    imgname = os.path.join(IMAGES_PATH,f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(imgname, frame)
    cv2.imshow('frame', frame)
    time.sleep(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```
### 1.2 Labelling 
In this stage, I used a library called labelme which allows you to draw bounding boxes around the object you want to detect (in this case a face) and 
returns the annotations as a json file.


<img width="1440" alt="Screenshot 2023-03-17 at 1 22 46 PM" src="https://user-images.githubusercontent.com/87010794/225891893-5fb96c6e-131a-4b57-b3d7-083853882b19.png">

## 2. Load Data and Create Image Loading Function

### 2.1 Tensorflow Pipeline
After collecting the images, I added them to my own TF pipeline using tf.data.

```python
images = tf.data.Dataset.list_files(('data/images/*.jpg'), shuffle=False)
```
### 2.2 Image Loading Function
The function load_image takes the image path and returns the image. After that I mapped the function to all the elements of the tensorflow dataset.

```python
def load_image(filename): 
    byte_img = tf.io.read_file(filename)
    img = tf.io.decode_jpeg(byte_img)
    return img
```
```python
images = images.map(load_image)
```
## 3. Apply Image Augmentation on Images and Labels using Albumentations

### 3.1 Setup Albumentations Transform Pipeline
First, I defined the albumentations pipeline by choosing the transformations I wanted to apply to my images and specifying the bounding box format.

```python
augmentor = alb.Compose([alb.RandomCrop(width=450, height=450), 
                         alb.HorizontalFlip(p=0.5), 
                         alb.RandomBrightnessContrast(p=0.2),
                         alb.RandomGamma(p=0.2), 
                         alb.RGBShift(p=0.2), 
                         alb.VerticalFlip(p=0.5)], 
                       bbox_params=alb.BboxParams(format='albumentations', 
                                                  label_fields=['class_labels']))
```
### 3.2 Build and Run Augmentation Pipeline

```python
for image in os.listdir(os.path.join('data','images')):
        img = cv2.imread(os.path.join('data','images', image))

        coords = [0,0,0.00001,0.00001]
        label_path = os.path.join('data', 'labels', f'{image.split(".")[0]}.json')
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                label = json.load(f)

            coords[0] = label['shapes'][0]['points'][0][0]
            coords[1] = label['shapes'][0]['points'][0][1]
            coords[2] = label['shapes'][0]['points'][1][0]
            coords[3] = label['shapes'][0]['points'][1][1]
            coords = list(np.divide(coords, [1280,720,1280,720]))

        try: 
            for x in range(60):
                augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
                cv2.imwrite(os.path.join('aug_data', 'images', f'{image.split(".")[0]}.{x}.jpg'), augmented['image'])

                annotation = {}
                annotation['image'] = image

                if os.path.exists(label_path):
                    if len(augmented['bboxes']) == 0: 
                        annotation['bbox'] = [0,0,0,0]
                        annotation['class'] = 0 
                    else: 
                        annotation['bbox'] = augmented['bboxes'][0]
                        annotation['class'] = 1
                else: 
                    annotation['bbox'] = [0,0,0,0]
                    annotation['class'] = 0 


                with open(os.path.join('aug_data', 'labels', f'{image.split(".")[0]}.{x}.json'), 'w') as f:
                    json.dump(annotation, f)

        except Exception as e:
            print(e)
```
This code runs through every image in the images folder and reads the image as a numpy array, then runs through every corresponding label in the labels 
folder and extracts the coordinates of the bounding box for each image. After that, it uses the augmentor function defined earlier and creates 60 new
augmented images per original image. Lastly, for each new augmented image, it creates a new annotation dictionary with the image filename, the class
(1 for 'face', 0 for no face) and the new augmented bounding box coordinates.
Thanks to this augmentation pipeline, I managed to grow my original image dataset from 90 to nearly 5000. WOW!

## 4 Load Augmented Data to Pipeline

### 4.1 Load and Rescale Images
At this point, I added my augmented images to my TF pipeline. I also resized the images to 120X120 and normalized pixel values.

```python
list_images = tf.data.Dataset.list_files('aug_data/images/*.jpg', shuffle=False)
list_images = list_images.map(load_image)
list_images = list_images.map(lambda x: tf.image.resize(x, (120,120)))
list_images = list_images.map(lambda x: x/255)
```
### 4.2 Create Label Loading Function and Load Labels
I defined a label loading function that takes the label path as input and loads the json file.

```python
def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding = "utf-8") as f:
        label = json.load(f)
        
    return [label['class']], label['bbox']
```
then added the labels to the TF pipeline.

```python]
list_labels = tf.data.Dataset.list_files('aug_data/labels/*.json', shuffle=False)
list_labels = list_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))
```
### 4.3 Combine images and labels
At this point, all I had to do was zip the images and labels and get to the final dataset.

```python
list_ds = tf.data.Dataset.zip((list_images, list_labels))
```

### 4.4 Split dataset into training, validation and testing datasets.
For deep learning, it is recommeneded to so.

```python
image_count = len(list_ds)
val_size = int(image_count * 0.3)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)
```
```python
val_image_count = len(val_ds)
test_size = int(val_image_count * 0.6)
val_ds = val_ds.skip(test_size)
test_ds = val_ds.take(test_size)
```
### 4.5 Batch, Shuffle and Prefetch datasets.
The last preprocessing step was to shuffle the datasets, group them into batches of 8 and add a call to prefetch. These strategies are meant to avoid 
bottlenecks while training the data.

```python
train_ds = train_ds.shuffle(len(train_ds))
train_ds = train_ds.batch(8)
train_ds = train_ds.prefetch(4)

val_ds = val_ds.shuffle(len(val_ds))
val_ds = val_ds.batch(8)
val_ds = val_ds.prefetch(4)

test_ds = test_ds.shuffle(len(test_ds))
test_ds = test_ds.batch(8)
test_ds= test_ds.prefetch(4)
```

## 5. Build Deep Learning Model Using Functional API
At this point, everything was ready to start defining the architecture of my deep learning model.

```python
def build_model(): 
    input_layer = Input(shape=(120,120,3))
    
    vgg = VGG16(include_top=False)(input_layer)

    # Classification Model  
    f1 = GlobalMaxPooling2D()(vgg)
    class1 = Dense(2048, activation='relu')(f1)
    class2 = Dense(1, activation='sigmoid')(class1)
    
    # Bounding box model
    f2 = GlobalMaxPooling2D()(vgg)
    regress1 = Dense(2048, activation='relu')(f2)
    regress2 = Dense(4, activation='sigmoid')(regress1)
    
    facetracker = Model(inputs=input_layer, outputs=[class2, regress2])
    return facetracker
```
This code uses the Model class from the functional API to define a custom neural network. 
This custom model is structured as follows:
- Input layer
- VGG16. The input layer is added to this pretrained CNN that does image classification. The parameter 'include_top=False' specifies that the prediction layers 
of the VGG16 are excluded.
- Two Dense layers with their own outputs: one for classifiying whether the image contains a face or not, and another for predicting the bounding box coordinates of the face.

## 6. Define losses and optimizer
The optimizer I chose was Adam and I defined two different losses for the two prediction heads: one for the classification and another one for the regression.
For the classification I chose BinaryCrossEntropy while for the regression I created a custom localization loss as follows:

```python
def localization_loss(y_true, yhat):            
    delta_coord = tf.reduce_sum(tf.square(y_true[:,:2] - yhat[:,:2]))
                  
    h_true = y_true[:,3] - y_true[:,1] 
    w_true = y_true[:,2] - y_true[:,0] 

    h_pred = yhat[:,3] - yhat[:,1] 
    w_pred = yhat[:,2] - yhat[:,0] 
    
    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true-h_pred))
    
    return delta_coord + delta_size
```
This function sums the squared difference between the true coordinates and the predicted ones, the squared differences between the true height and predicted height
of the bounding box, and the squared difference between the true and predicted width of the bounding box.

## 7. Train the Model
In order to train the model, I defined a custom training class that subclasses the Model class from the Keras Functional API. This allowed me to create a 
customized training pipeline where I could add my custom model and my custom loss functions.

```python
class FaceTracker(Model): 
    def __init__(self, facetracker,  **kwargs): 
        super().__init__(**kwargs)
        self.model = facetracker

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt
    
    def train_step(self, batch, **kwargs): 
        
        X, y = batch
        
        with tf.GradientTape() as tape: 
            classes, coords = self.model(X, training=True)
            
            batch_classloss = self.closs(y[0], classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
            
            total_loss = batch_localizationloss+0.5*batch_classloss
            
            grad = tape.gradient(total_loss, self.model.trainable_variables)
        
        opt.apply_gradients(zip(grad, self.model.trainable_variables))
        
        return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_localizationloss}
    
    def test_step(self, batch, **kwargs): 
        X, y = batch
        
        classes, coords = self.model(X, training=False)
        
        batch_classloss = self.closs(y[0], classes)
        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_localizationloss+0.5*batch_classloss
        
        return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_localizationloss}
        
    def call(self, X, **kwargs): 
        return self.model(X, **kwargs)
```
After this, I instantiated the model, compiled it and then called fit to start training.

```python
model = FaceTracker(facetracker)
model.compile(opt, classloss, regressloss)
hist = model.fit(train_ds, epochs=40, validation_data=val_ds, callbacks=[tensorboard_callback])
```
## 8. Plot Performance
After training, I plotted the total loss, classification loss and localization loss per each epoch.

<img width="773" alt="Screenshot 2023-03-17 at 3 41 33 PM" src="https://user-images.githubusercontent.com/87010794/225921568-fd05ec9e-4f2f-459c-a7ea-bbd3da1e7b3a.png">

## 9. Make Predictions
At this point, I made predictions on the test set.

```python
test_data = test_ds.as_numpy_iterator()
test_sample = test_data.next()
yhat = facetracker.predict(test_sample[0])

fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx in range(4): 
    sample_image = test_sample[0][idx]
    sample_coords = yhat[1][idx]
    
    if yhat[0][idx] > 0.9:
        cv2.rectangle(sample_image, 
                      tuple(np.multiply(sample_coords[:2], [120,120]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [120,120]).astype(int)), 
                            (255,0,0), 2)
    
    ax[idx].imshow(sample_image)
```

<img width="767" alt="Screenshot 2023-03-17 at 3 44 30 PM" src="https://user-images.githubusercontent.com/87010794/225922365-ea6b5d61-c2d9-413a-9ef0-bec7ea925f03.png">

## 10. Save Model

```python
facetracker.save('facetracker.h5')
facetracker = load_model('facetracker.h5')
```
## 11. Real Time Detection

```python
cap = cv2.VideoCapture(0)
while cap.isOpened():
    _ , frame = cap.read()
    frame = frame[50:500, 50:500,:]
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120,120))
    
    yhat = facetracker.predict(np.expand_dims(resized/255,0))
    sample_coords = yhat[1][0]
    
    if yhat[0] > 0.5: 
        # Controls the main rectangle
        cv2.rectangle(frame, 
                      tuple(np.multiply(sample_coords[:2], [450,450]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [450,450]).astype(int)), 
                            (255,0,0), 2)
        # Controls the label rectangle
        cv2.rectangle(frame, 
                      tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int), 
                                    [0,-30])),
                      tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                    [80,0])), 
                            (255,0,0), -1)
        
        # Controls the text rendered
        cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
                                               [0,-5])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
    cv2.imshow('FaceTrack', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
```


<img width="576" alt="Screenshot 2023-03-17 at 4 04 12 PM" src="https://user-images.githubusercontent.com/87010794/225927899-9786d695-6790-4c49-ab3b-d23488a12290.png">


    
