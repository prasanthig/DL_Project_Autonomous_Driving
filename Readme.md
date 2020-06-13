# Deep Learning Project - Team 22
This  project  proposes  a  method  to  reconstructa  semantic  top  view  layout  of  an  autonomous driving  scene,  given  six  different  surrounding images captured from the driving platform. The objective can be split into two tasks: to generate the road map layout and to detect other objects like  cars,  trucks,  pedestrians,  etc  in  the  scene. The road map layout is generated using a vanilla autoencoder  and  the  objects  in  the  scene  are detected using a modified YOLOv3 architecture for oriented objects. We obtained an accuracy of 0.77 for the road map layout task and an accuracy of 0.016 for the object detection task.


## STEPS:
1. Place data in data folder
2. To run the Roadmap layout generation
    ```
    cd ./Roadmap
    python main.py --data_dir='../data' --batch_size=4 --epochs=10 
    ```
   The model is saved in ./Models folder. The images generated from the validation set is stored in TestImages folder. Tensorflow logging is saved in runs folder
    
3. To run Bounding box detection
   ``` 
   cd ./BoundingBoxDetection 
   ## To train model
   python main.py --model='/path/to/pretrained model' --data_dir='../data' 
   ## To detect objects
   python detect.py --data_dir='../data' --checkpoint_model='/path/to/trained_model/'
   ```
   The model is saved in ./Models folder. Tensorflow logging is saved in runs folder. The output from detect.py is stored in outputs folder.
