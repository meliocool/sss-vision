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
import glob
import requests

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

S_Units = {
    "Acid Angel from Asia": ["S2 Hyerin", "S5 YooYeon", "S7 NaKyoung", "S8 YuBin"],
    "KRystal Eyes": ["S1 SeoYeon", "S3 Jiwoo", "S4 ChaeYeon", "S6 SooMin"],
    "Acid Eyes": ["S1 SeoYeon","S2 Hyerin", "S3 Jiwoo", "S4 ChaeYeon", 
                  "S5 YooYeon", "S6 SooMin", "S7 NaKyoung", "S8 YuBin"],
    "LOVElution": ["S1 SeoYeon", "S2 Hyerin", "S8 YuBin", "S9 Kaede", 
                   "S10 Dahyun", "S13 Nien", "S14 SoHyun", "S15 Xinyu"],
    "EVOLution": ["S3 Jiwoo", "S4 ChaeYeon", "S5 YooYeon", "S6 SooMin", 
                  "S7 NaKyoung", "S11 Kotone", "S12 YeonJi", "S16 Mayu"],
    "NXT": ["S17 Lynn", "S18 JooBin", "S19 HaYeon", "S20 ShiOn"],
    "Aria": ["S3 Jiwoo", "S4 ChaeYeon", "S9 Kaede", "S10 Dahyun", "S13 Nien"],
    "Glow": ["S21 Chaewon", "S22 Sullin", "S23 SeoAh", "S24 JiYeon"],
    "Visionary Vision": ["S2 Hyerin", "S5 YooYeon", "S7 NaKyoung", "S8 YuBin", "S9 Kaede",
                         "S11 Kotone", "S12 YeonJi", "S13 Nien", "S14 SoHyun", "S15 Xinyu",
                         "S17 Lynn", "S24 JiYeon"],
    "Hatchi": ["S3 Jiwoo", "S4 ChaeYeon", "S6 SooMin", "S5 YooYeon", "S11 Kotone", 
               "S16 Mayu", "S20 ShiOn", "S21 Chaewon"]
}

faceModel_Path = os.path.join('.', 'Face-Embeddings', 'dimensionFace_trainedV2.npy')
name_model_path = os.path.join('.', 'Face-Embeddings', 'dimensionName_trainedV2.npy')

faceModel = np.load(faceModel_Path)
nameModel = np.load(name_model_path)

directory_face_region = os.path.join('..', 'SimpleProjects', 'is_it_an_S', 'training_images')
directory_input_save = os.path.join('.', 'static', 'scanned-images')
directory_nonMember = os.path.join('..', 'memberFalse')

detector = MTCNN()
MLmodel = InceptionResnetV1(pretrained='vggface2').eval()

def get_image_count(folder):
    folder_path = os.path.join(directory_face_region, folder)
    image_files = glob.glob(os.path.join(folder_path, "*.jpg"))
    return len(image_files)

def save_face(face, folder, count):
    folder_path = os.path.join(directory_face_region, folder)
    pil_face = Image.fromarray(cv.cvtColor(face, cv.COLOR_BGR2RGB))
    save_path = os.path.join(folder_path, f'{folder}_face_{count}.jpg')
    pil_face.save(save_path)

def uploaded_pic(img, folder, count):
    if folder == "Not a tripleS member":
        path = directory_nonMember
    else:
        path = os.path.join(directory_input_save, folder)

    save_path = os.path.join(path, f'{folder}_{count}.jpg')
    with open(save_path, 'wb') as image_file:
        image_file.write(img.getvalue())

def text_results(max_font_scale, rect_width, text, font, thickness=1):
    font_scale = max_font_scale
    while font_scale > 0.5: 
        text_size, _ = cv.getTextSize(text, font, font_scale, thickness)
        text_width = text_size[0]
        if text_width <= rect_width: 
            return font_scale
        font_scale -= 0.1  
    return 0.5 

def extract_s_num(name):
    try:
        return int(name.split(' ')[0][1:])
    except(ValueError, IndexError):
        return None

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/gallery')
def gallery():
    mem = os.listdir(directory_input_save)
    content = {}
    members = [m for m in mem if extract_s_num(m) is not None]
    subU = [m for m in mem if extract_s_num(m) is None]
    mem_sort = sorted(members, key=extract_s_num)
    subU_sort = sorted(subU)
    mem_sorted = mem_sort + subU_sort
    for member in mem_sorted:
        path = os.path.join(directory_input_save, member)
        if os.path.isdir(path):
            image = [f for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            content[member] = image
    return render_template('gallery.html', content = content)

@app.route('/upload', methods=['POST'])
def image_analysis():
    global S_detected
    S_detected = []
    new_count = 1 # image counter
    file = request.files.get('image')
    image_url = request.form.get('image_url')
    if file and file.filename != '':
        image = np.frombuffer(file.read(), np.uint8)
    elif image_url:
        response = requests.get(image_url)
        if response.status_code == 200:
            image = np.frombuffer(response.content, np.uint8)
        else:
            return "Error unable to download image from URL", 400
    else:
        return "No image inserted", 400
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
            similarity = cosine_similarity([embedding], [stored_embedding])[0][0] # 1 min video of cosine similarity btw
        
            if similarity > max_similarity:
                max_similarity = similarity
                match = idx

        similarity_threshold = 0.7 # can be adjusted

        if max_similarity > similarity_threshold:
            name = sss[nameModel[match]]
            confidence_text = f'{round(max_similarity * 100)}% Confident'
            current_count = get_image_count(name)
            new_count = current_count + 3 # asspull number btw fck this
            S_detected.append(name)  # append anyway
            save_face(facialRegion, name, new_count)
        else:
            name = "Not a tripleS member"
            confidence_text = ""

        fit_name = text_results(0.9, width, name, cv.QT_FONT_NORMAL, thickness=2)
        fit_conf = text_results(0.9, width, confidence_text, cv.QT_FONT_NORMAL, thickness=2)

        Scolor = S_Color.get(name, (0, 0, 255))

        cv.rectangle(img, (x, y), (x+width, y+height), Scolor, 2)
        cv.putText(img, name, (x, y - 10), cv.QT_FONT_NORMAL, fit_name, Scolor, 2)
        cv.putText(img, confidence_text, (x, y + height + 20), cv.QT_FONT_NORMAL, fit_conf, Scolor, 2)
            
    output_width = 500 
    output_height = int((output_width / img.shape[1]) * img.shape[0])
    img_resized = cv.resize(img, (output_width, output_height))

    img_rgb_final = cv.cvtColor(img_resized, cv.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb_final)

    img_io = io.BytesIO()
    pil_img.save(img_io, 'JPEG', quality=100)
    img_io.seek(0)

    member_count = len(S_detected)
    unit = False
    for subU, mem in S_Units.items():
        if all(s in S_detected for s in mem):
            name = subU
            unit = True
            break
    if unit == False:
        if member_count == 2:
            uploaded_pic(img_io, "Duo", new_count)
        elif member_count == 3:
            uploaded_pic(img_io, "Trio", new_count)
        elif member_count > 3:
            uploaded_pic(img_io, "Group Picture", new_count)
    
    uploaded_pic(img_io, name, new_count)

    return send_file(img_io, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)