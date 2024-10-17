import os
import cv2 as cv
import numpy as np
from mtcnn import MTCNN
import torch
from facenet_pytorch import InceptionResnetV1

sss = ['S1 SeoYeon', 'S2 Hyerin', 'S3 Jiwoo', 'S4 ChaeYeon', 'S5 YooYeon', 'S6 SooMin', 'S7 NaKyoung',
       'S8 YuBin', 'S9 Kaede', 'S10 Dahyun', 'S11 Kotone', 'S12 YeonJi', 'S13 Nien', 'S14 SoHyun',
       'S15 Xinyu', 'S16 Mayu', 'S17 Lynn', 'S18 JooBin', 'S19 HaYeon', 'S20 ShiOn', 'S21 Chaewon',
       'S22 Sullin', 'S23 SeoAh', 'S24 JiYeon']

training_images = "C:\\Users\\Asus VivobookPro\\Documents\\CODING STUFF\\AI\\SimpleProjects\\is_it_an_S\\training_images"

detector = MTCNN()
MLmodel = InceptionResnetV1(pretrained='vggface2').eval()

sFace = []
sNames = []

def training():
    for Pic in sss:
        path = os.path.join(training_images, Pic) 
        label = sss.index(Pic)
        for IMG in os.listdir(path):
            img_path = os.path.join(path, IMG)
            img_array = cv.imread(img_path)
            img_rgb = cv.cvtColor(img_array, cv.COLOR_BGR2RGB)
            faces = detector.detect_faces(img_rgb)
            for face in faces:
                x, y, width, height = face['box']
                facialRegion = img_rgb[y:y+height, x:x+width]
                facialRegion_resized = cv.resize(facialRegion, (160, 160))
        
                facialRegion_tensor = torch.tensor(facialRegion_resized).float().permute(2, 0, 1) / 255.0
                facialRegion_tensor = facialRegion_tensor.unsqueeze(0)
                
                with torch.no_grad():
                    embedding = MLmodel(facialRegion_tensor)
                
                sFace.append(embedding.squeeze().numpy())
                sNames.append(label)

training()
print("Finished learning, I know tripleS members now!")

sFace = np.array(sFace)
sNames = np.array(sNames)

np.save('dimensionFace_trainedV2.npy', sFace)
np.save('dimensionName_trainedV2.npy', sNames)