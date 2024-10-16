﻿SSS-Vision
=  

## NOT IN A VIRTUAL ENVIRONMENT, YOU NEED TO INSTALL THE LIBRARIES USED IN THIS PROJECT

## Overview  
SSS-Vision is a web-based, flask powered
AI system that could receive an image of tripleS members from the user and return a face-recognized and labeled image. The model uses MTCNN and Facenet pretrained model to recognize the face, and i'm using a python-based web-scraper to get the data (pictures of tripleS members) from Pinterest.

## Small Note  
opencv2 FaceLBPHRecognizer (haarcascade_frontalface.xml) using only 2 members, S1 SeoYeon and S7 NaKyoung turned out great but it struggles with side-profiles so it is replaced with mtcnn for the face recognition


## Library usage:  
- **flask** -> for the web server (request and send_file is like the name suggest)  
- **mtcnn** -> to perform face detection in images  
- **cv2** -> image reading, recoloring, and resizing for training purposes    
- **numpy** -> to save the embedded faces, numpy array manipulation, etc    
- **torch** -> to manipulate the facial region of the image into tensors to prepare for training    
- **facenet/inceptionresnetv1** -> the base pretrained model to compare the faces   
- **cosine_similarity** -> imported from sklearn.metrics.pairwise to calculate the similarity of the tensored face and the stored face embeddings  
- **Image** -> imported from PIL for impace processing to be sent from the model to the front-end  
- **io** -> same as Image  
- **os** -> to go into every folder and join directories for training purposes    
- **glob** -> honestly damn near useless, its to just find out how many images are in the folder 


## References:  
- **Prototyping** = https://youtu.be/oXlwWbU8l2o?si=XJi3nVT9YhfxdC38&t=10146  
- **Flask** = https://youtu.be/Z1RJmh_OqeA?si=BJCojDmc5RgcpbGR  
- **HTML CSS** = Basic is from brocode, and past projects with BNCC  
- **JavaScript** = AJAX tutorial from BNCC, and ChatGPT  
- **Neural Network stuff** = https://youtu.be/-rrxxpiZa00?si=h5BGQWerr0EDCQ31  

## Thanks for reading!

