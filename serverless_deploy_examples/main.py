# Python packages
from base64 import b64decode
from io import BytesIO
from os.path import isfile

# Packages from pip
import numpy as np
from numpy import asarray, expand_dims
from PIL import Image
import cv2
import requests
from google.cloud import storage

# Package in current directory
from Mask_RCNN.mrcnn.config import Config
from Mask_RCNN.mrcnn.model import MaskRCNN
from Mask_RCNN.mrcnn.utils import Dataset
from Mask_RCNN.mrcnn.utils import compute_ap
from Mask_RCNN.mrcnn.model import load_image_gt
from Mask_RCNN.mrcnn.model import mold_image

class FishConfig(Config):
	NAME = "fish_cfg"
	NUM_CLASSES = 1 + 1

class InferenceConfig(FishConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class FishDataset(Dataset):
  def load_dataset(self, dataset_dir, is_train=True):
    self.add_class("dataset", 1, "Fish")

#Função para segmentar a régua e calcular a quantidade de pixels

if(not isfile("/tmp/maskrcnn.h5")):
  storage_client = storage.Client.from_service_account_json('even-gearbox-325502-5575649cd2f4.json')
  bucket_maskrcnn = storage_client.get_bucket("maskrcnn_test-server")
  file_mask = bucket_maskrcnn.get_blob("maskrcnn.h5")
  file_mask.download_to_filename("/tmp/maskrcnn.h5")

inference_config = InferenceConfig()

model = MaskRCNN(mode="inference", 
                 config=inference_config,
                 model_dir='Resultados/')

model.load_weights('/tmp/maskrcnn.h5', by_name=True)

test = FishDataset()
test.load_dataset('Mask_RCNN/Fish', is_train=True)
test.prepare()

def inference(request):
  img = Image.open(BytesIO(b64decode(request.form['image_base64'])))

  image = np.array(img)
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  
  #rotacionar a imagem para o peixe ficar na horizontal
  if(image.shape[0] > image.shape[1]):
    image = cv2.transpose(image)
  
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  
  kernel = np.ones((5, 5), np.uint8)

  #Limites para encontrar a régua pela sua cor
  lower = np.array([160, 70, 55])
  upper = np.array([176, 255, 255])

  mask = cv2.inRange(hsv, lower, upper)
  
  (cnts, _) = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  #Ordenando decrescente os contornos pelas áreas 
  cnt = sorted(cnts, key = cv2.contourArea, reverse = True)
  
  #Verificando se realmente é uma régua pela quantidade de pontas
  #Uma régua, como é um retângulo, tem 4 pontas, mas colocou como 10 como margem de erro 
  #Pegando o primeiro contorno que se encaixa nos critérios
  for c in cnt:
    per = 0.015 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, per, True)
    if(len(approx) <= 10):
      break
  
  left = tuple(c[c[:, :, 0].argmin()][0])[0]
  right = tuple(c[c[:, :, 0].argmax()][0])[0]
  
  results = model.detect([image], verbose=0)
  r = results[0]

  tamanho = 0

  for count, box in enumerate(r['rois']):
    (y1, x1, y2, x2) = box
    tamanho = float("{0:.2f}".format((((x2 - x1) * 31) / (right - left))))
  
  response = str({'Tamanho': tamanho})

  headers = {
          'Access-Control-Allow-Origin': '*'
  }

  return (response, 200, headers)
