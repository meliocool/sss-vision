# SSS-Vision: AI-Powered Face Recognition App

A web application that uses machine learning to identify members of the K-Pop group tripleS from an uploaded photo.

---

### Live Demo

Since hosting the full ML model is resource-intensive, here is a complete video demonstration of the working application.

[![SSS-Vision Demo Video](https://github.com/meliocool/sss-vision/blob/main/image.png?raw=true)](https://youtu.be/ZcOA5ZKQk_o)  
**(Click the image to watch the full video demo on YouTube)**

### Key Features

* **AI-Powered Recognition:** Utilizes a custom-trained face recognition model to make predictions.
* **Dynamic Gallery:** Displays a history of uploaded images and their predictions (runs locally).
* **Interactive UI:** A simple and clean user interface built with HTML, CSS, and JavaScript.

### Tech Stack

* **Backend:** Python, Flask, NumPy
* **Frontend:** HTML5, CSS3, JavaScript (with AJAX)
* **Machine Learning:** MTCCN, cv2, torch, numpy, inceptionresnetv1, cosine_similarity

### How to Run Locally

1.  Clone the repository: `git clone https://github.com/meliocool/sss-vision.git`
2.  Create and activate a virtual environment.
3.  Install dependencies: `pip install -r requirements.txt`
4.  Run the application: `python sss-vision.py`
5.  Open your browser to `http://127.0.0.1:5000`

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
