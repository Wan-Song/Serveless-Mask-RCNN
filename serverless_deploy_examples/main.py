# Python's standard libraries 
from base64 import b64decode
from io import BytesIO
from os.path import isfile
from math import ceil, sqrt

# Packages from pip
import numpy as np                              #pip install numpy
from PIL import Image                           #pip install Pillow
import cv2                                      #pip install opencv-python
import requests                                 #pip install requests
from google.cloud import storage                #pip install google-cloud-datastore

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

#Verificando se já baixou o modelo treinado
if(not isfile("/tmp/maskrcnn.h5")):
  storage_client = storage.Client.from_service_account_json('even-gearbox-325502-5575649cd2f4.json')
  bucket_maskrcnn = storage_client.get_bucket("maskrcnn_test-server")
  file_mask = bucket_maskrcnn.get_blob("maskrcnn.h5")
  file_mask.download_to_filename("/tmp/maskrcnn.h5")

#Preparando o modelo para efetuar as previsões e carregando o modelo treinado
inference_config = InferenceConfig()
model = MaskRCNN(mode="inference", 
                 config=inference_config,
                 model_dir='Resultados/')
model.load_weights('/tmp/maskrcnn.h5', by_name=True)

#Criando a instância do modelo com os pesos treinados
test = FishDataset()
test.load_dataset('Mask_RCNN/Fish', is_train=True)
test.prepare()


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#Função responsável por encontrar a régua por meio da cor (Rosa)
def ruler(img):
  #Convertendo a imagem para base de cor HSV
  hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

  kernel = np.ones((5, 5), np.uint8)
  
  mask_ruler = np.zeros((img.shape[0], img.shape[1]), np.uint8)

  #Limites inferior e superior para encontrar a régua pela sua cor
  lower = np.array([160, 70, 55])
  upper = np.array([176, 255, 255])

  mask = cv2.inRange(hsv, lower, upper)
  (cnts, _) = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  #Ordenando decrescente os contornos pelas áreas 
  cnt = sorted(cnts, key = cv2.contourArea, reverse = True)
  
  #Caso a régua não seja encontrada
  if(len(cnt) == 0):
    print("ERRO! Régua não foi encontrada.")
    return mask_ruler, 0

  #Teste para verificar qual contorno encontrado é da régua
  for c in cnt:
    per = 0.015 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, per, True)
    if(len(approx) <= 10):
      break

  #Desenhando somente a régua na imagem
  mask_ruler = cv2.drawContours(mask_ruler, [c], -1, 255, -1)

  return mask_ruler, 1


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#Função responsável por remover os ruídos da máscara
def remove_noise(img):

  #Encontrando a posição do peixe
  (cnts, _) = cv2.findContours(img, cv2.RETR_TREE,
                               cv2.CHAIN_APPROX_SIMPLE)
  #Ordenando os pontos encontrados e pegando somente o maior
  cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

  #Zerando a imagem
  img[:] = 0

  #Desenhando somente o maior objeto na imagem novamente
  img = cv2.drawContours(img, [cnt], -1, (255, 255, 255), -1)

  return img


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#Função para deixar a imagem com o rabo na esquerda
def invert_image(img):

  #Largura da imagem
  width = img.shape[1] 

  #Encontrando a posição do peixe
  (cnts, _) = cv2.findContours(img, cv2.RETR_TREE, 
                              cv2.CHAIN_APPROX_SIMPLE)  
  #Ordenando os pontos encontrados e pegando somente o maior
  cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

  #Pegando a primeira e a última posição x da máscara do peixe
  leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])[0]
  rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])[0]

  #Pegando parte da cabeça e do rabo do peixe para descobrir a posição de cada um
  img_left = img[:, leftmost:leftmost + ceil(width * 0.20)]
  img_right = img[:, rightmost - ceil(width * 0.20):rightmost]

  #Calculando a quantidade de pixels de cada parte para usar como indicador para 
  #descobrir o lado da cabeça e do rabo
  count_left = cv2.countNonZero(img_left)
  count_right = cv2.countNonZero(img_right)

  #O lado que tiver mais pixels será designado como sendo a cabeça
  #Se a cabeça estiver no lado esquerdo, gira a imagem para deixar no lado direito
  if(count_left > count_right):
    img = cv2.rotate(img, cv2.ROTATE_180)

  #Encontrando a posição do peixe
  (cnts, _) = cv2.findContours(img, cv2.RETR_TREE, 
                              cv2.CHAIN_APPROX_SIMPLE)
  #Ordenando os pontos encontrados e pegando somente o maior
  cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

  #Pegando 20% da parte do rabo para calcular o padrão e furcal
  leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])[0]
  img_tail = img[:, 0:leftmost + ceil(width * 0.20)]

  return img, img_tail


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#Função utilizada para encontrar o ponto inicial para calcular o tamanho padrão
def calculate_padrao(img_tail):

  #Posição para calcular o tamanho padrão
  position = -1

  #Lista com o valor da quantidade de pixels brancosde cada coluna
  white_column = []

  #Encontrando a posição da máscara do rabo
  (cnts, _) = cv2.findContours(img_tail, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  #Ordenando os pontos encontrados e pegando somente o maior
  cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

  #Pegando a primeira e a última posição x da máscara
  leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])[0]
  rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])[0]

  #Largura total da máscara do rabo
  width_mask = rightmost - leftmost
  #Proporção para a quantidade de pontos
  proportion = width_mask // 50

  #Calculando a quantidade de pixels brancos para cada coluna
  for x in range(leftmost, rightmost + 1):
    search = np.where(img_tail[:, x] == 255)[0]
    white_column.append(len(search))

  previous = 0
  count = 0

  for x in range(len(white_column) - 1, -1, -1):
    if(white_column[x] >= previous):
      count = count + 1
      position = x + leftmost

      if(count == proportion):
        break
   
    else:
      count = 0
    
    previous = white_column[x]

  return position

  
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#Função utilizada para encontrar o ponto inicial para calcular o tamanho furcal
def calculate_furcal(img_tail):

  position = -1

  #Encontrando a posição do peixe
  (cnts, _) = cv2.findContours(img_tail, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  #Ordenando os pontos encontrados e pegando somente o maior
  cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0] 

  #Pegando a primeira e a última posição x da máscara
  first = tuple(cnt[cnt[:,:,0].argmin()][0])[0]
  last = tuple(cnt[cnt[:,:,0].argmax()][0])[0]
  
  flag = False

  for x in range(first, last):
    #Procurando as posições na coluna x que possuem o valor 255 (Branco)
    search = np.where(img_tail[:, x] == 255)[0]

    upper_limit = search[0]
    under_limit = search[-1]

    search = np.where(img_tail[upper_limit:under_limit, x] == 0)[0]
    
    if(flag == False and len(search) > 0):
      flag = True
    elif(flag and len(search) == 0):
      position = x
      break
    
  return position
  

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#Função responsável por gerar a linha central do tamanho furcal e padrão
#e calcular os seus tamanhos
def calculate_sizes(img, points_fish, pos, width_ruler):
  size = 0

  if(pos == -1):
    return size

  search = np.where(img[:, pos] == 255)[0]

  upper_limit = search[0]
  under_limit = search[-1]

  middle = (upper_limit + under_limit) // 2

  p = points_fish.copy()
  del p[0:2]
  p.insert(0, (pos, middle))

  size_pixels = full_size(p)

  #Calculando o tamanho furcal do peixe
  size = (31 * size_pixels) / width_ruler
  size = float("{0:.2f}".format(size))

  return size


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#Função para calcular a distância dos pontos na lista
def full_size(p):
  size = 0
  for x in range(1, len(p)):
    size += sqrt((p[x][0] - p[x - 1][0]) ** 2 + (p[x][1] - p[x - 1][1]) ** 2)

  return size


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#Função para calcular a linha central da máscara
def centerline(img):

  #Posições de cada linha vertical
  percentage = [0, 0.15, 0.60, 0.80, 0.85, 0.9, 0.95, 1];

  #Vetor dos pontos para a linha central
  line = []

  #Encontrando a posição da máscara
  (cnts, _) = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  #Ordenando os pontos encontrados e pegando somente o maior
  cnt = sorted(cnts, key = cv2.contourArea, reverse = True)[0] 

  #Pegando a primeira e a última posição x (horizontal) da máscara
  leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])[0]
  rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])[0]

  #Largura total somente da máscara
  width_total = rightmost - leftmost

  #Calculando os pontos da linha central
  for p in percentage:
    x = leftmost + ceil(width_total * p)

    #Procurando as posições na coluna x que possuem o valor 255 (Branco)
    search = np.where(img[:, x] == 255)[0]

    #Encontrando o ponto máximo e mínimo na coluna
    y_top = search[0]
    y_bottom = search[-1]
    
    #y é o meio da máscara na posição x
    y = (y_top + y_bottom) // 2

    #Adciona o ponto na lista
    line.append((x, y))

  return line


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#Função principal que recebe a imagem
def inference(request):
  headers = {
          'Access-Control-Allow-Origin': '*'
  }

  #Imagem inicial
  img = Image.open(BytesIO(b64decode(request.form['image_base64'])))

  #Transformando a imagem em um numpy array
  image = np.array(img)
  
  #Rotacionando a imagem, caso seja necessário, para o peixe ficar na horizontal
  if(image.shape[0] > image.shape[1]):
    image = cv2.transpose(image)

  #Efetuando a previsão da  posição do peixe utilizando o modelo treinado
  results = model.detect([image], verbose=0)
  r = results[0]

  #Pegando a máscara somente com o peixe 
  (h, w, d) = r['masks'].shape
  mask_fish = np.zeros((h, w), np.uint8)
  #Gerando a máscara do peixe
  for i in range(d):
    mask = r['masks'][:, :, i]
    mask_fish = mask_fish | mask
  mask_fish = mask_fish * 255

  #Gerando a máscara da régua
  mask_ruler, flag = ruler(image) 

  #Caso tenha ocorrido erro para segmentar a régua
  if(flag == False):
      response = str({'Tamanho Total':0, 'Tamamho Furcal':0, 
                  'Tamanho Padrão':0, 'Observação':'Régua não encontrada'})
      return (response, 200, headers) 

  #Removendo os ruídos
  mask_fish = remove_noise(mask_fish)

  #Deixando a imagem com o rabo na esquerda
  (mask_fish, mask_tail) = invert_image(mask_fish)

  #Calculando o ponto para o tamanho furcal e padrão
  position_furcal = calculate_furcal(mask_tail)
  position_padrao = calculate_padrao(mask_tail)

  #Encontrando os pontos para a linha central do peixe
  points_fish = centerline(mask_fish)
  #Obtendo a largura do peixe
  full_fish_size = full_size(points_fish)

  #Encontrando os pontos para a linha central da régua
  points_ruler = centerline(mask_ruler)
  #Obtendo a largura da régua
  width_ruler = full_size(points_ruler)

  #Calculando o tamanho total do peixe
  size_fish = (31 * full_fish_size) / width_ruler
  size_fish = float("{0:.2f}".format(size_fish))

  #Calculando o tamanho furcal
  furcal_size = calculate_sizes(mask_fish, points_fish, position_furcal, width_ruler)
  #Calculando o tamanho padrão
  padrao_size = calculate_sizes(mask_fish, points_fish, position_padrao, width_ruler)

  response = str({'Tamanho Total':size_fish, 'Tamamho Furcal':furcal_size, 
                  'Tamanho Padrão':padrao_size, 'Observação':'Completo'})

  return (response, 200, headers)
