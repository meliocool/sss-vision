from flask import Flask, render_template, request, send_file
from mtcnn import MTCNN
import cv2 as cv
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import io
import os

app = Flask(__name__)

sss = ['S1 SeoYeon', 'S2 Hyerin', 'S3 Jiwoo', 'S4 ChaeYeon', 'S5 YooYeon', 'S6 SooMin', 'S7 NaKyoung',
       'S8 YuBin', 'S9 Kaede', 'S10 Dahyun', 'S11 Kotone', 'S12 YeonJi', 'S13 Nien', 'S14 SoHyun',
       'S15 Xinyu', 'S16 Mayu', 'S17 Lynn', 'S18 JooBin', 'S19 HaYeon', 'S20 ShiOn', 'S21 Chaewon',
       'S22 Sullin', 'S23 SeoAh', 'S24 JiYeon']

S_Color = {
    "S1 SeoYeon": (255, 144, 30),
    "S2 Hyerin": (236, 114, 145),
    "S3 Jiwoo": (51, 250, 250),
    "S4 ChaeYeon": (57, 199, 140),
    "S5 YooYeon": (208, 211, 242),
    "S6 SooMin": (174, 138, 227),
    "S7 NaKyoung": (160, 158, 95),
    "S8 YuBin": (225, 228, 255),
    "S9 Kaede": (51, 204, 255),
    "S10 Dahyun": (227, 160, 251),
    "S11 Kotone": (0, 223, 255),
    "S12 YeonJi": (225, 105, 65),
    "S13 Nien": (67, 163, 255),
    "S14 SoHyun": (166, 52, 16),
    "S15 Xinyu": (21, 8, 200),
    "S16 Mayu": (137, 160, 255),
    "S17 Lynn": (187, 85, 153),
    "S18 JooBin": (102, 230, 164),
    "S19 HaYeon": (182, 209, 120),
    "S20 ShiOn": (138, 74, 246),
    "S21 Chaewon": (236, 224, 222),
    "S22 Sullin": (143, 188, 143),
    "S23 SeoAh": (235, 206, 135),  
    "S24 JiYeon": (85, 181, 255)
}

faceModel = np.load('C:\\Users\\Asus VivobookPro\\Documents\\CODING STUFF\\AI\\SSS-Vision\\Face-Embeddings\\dimensionFace_trainedV2.npy')
nameModel = np.load('C:\\Users\\Asus VivobookPro\\Documents\\CODING STUFF\\AI\\SSS-Vision\\Face-Embeddings\\dimensionName_trainedV2.npy')
directory = 'C:\\Users\\Asus VivobookPro\\Documents\\CODING STUFF\\AI\\SimpleProjects\\is_it_an_S\\training_images'
count_dict = {name: 0 for name in sss}

detector = MTCNN()
MLmodel = InceptionResnetV1(pretrained='vggface2').eval()

def save_face(face, folder, count):
    folder_path = os.path.join(directory, folder)
    pil_face = Image.fromarray(cv.cvtColor(face, cv.COLOR_BGR2RGB))
    save_path = os.path.join(folder_path, f'{folder}_face_{count}.jpg')
    pil_face.save(save_path)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/upload', methods=['POST'])
def image_analysis():
    file = request.files['image']
    image = np.frombuffer(file.read(), np.uint8)
    img = cv.imdecode(image, cv.IMREAD_COLOR)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)
    for face in faces:
        x, y, width, height = face['box']
        facialRegion = img_rgb[y:y+height, x:x+width]

        facialRegion_resized = cv.resize(facialRegion, (160, 160))
        facialRegion_tensored = torch.tensor(facialRegion_resized).float().permute(2, 0, 1) / 255
        facialRegion_tensored = facialRegion_tensored.unsqueeze(0)

        with torch.no_grad():
            embedding = MLmodel(facialRegion_tensored).numpy().flatten()

        max_similarity = -1
        match = -1

        for idx, stored_embedding in enumerate(faceModel):
            similarity = cosine_similarity([embedding], [stored_embedding])[0][0]
        
            if similarity > max_similarity:
                max_similarity = similarity
                match = idx

        similarity_threshold = 0.7

        if max_similarity > similarity_threshold:
            name = sss[nameModel[match]]
            confidence_text = f'{round(max_similarity * 100)}% Confident'
            count_dict[name] += 1
            save_face(facialRegion, name, count_dict[name])
        else:
            name = "Not a tripleS member"
            confidence_text = ""
        
        Scolor = S_Color.get(name, (0, 255, 0))
        cv.rectangle(img, (x, y), (x+width, y+height), Scolor, 2)
        cv.putText(img, name, (x, y - 10), cv.QT_FONT_NORMAL, 0.9, Scolor, 2)
        cv.putText(img, confidence_text, (x, y + height + 20), cv.QT_FONT_NORMAL, 0.9, Scolor, 2)
            
    output_width = 500 
    output_height = int((output_width / img.shape[1]) * img.shape[0])
    img_resized = cv.resize(img, (output_width, output_height))

    img_rgb_final = cv.cvtColor(img_resized, cv.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb_final)

    img_io = io.BytesIO()
    pil_img.save(img_io, 'JPEG', quality=100)
    img_io.seek(0)

    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)