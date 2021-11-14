# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 21:54:45 2021

@author: pcmaroc
"""
import dlib 
import cv2
import numpy as np 
import time
from emoji import emojize
start_time = time.time()




def extract_index(nparray):
    index = None 
    for num in nparray[0]:
        index = num
        break
    return index 
        


imgSrc = cv2.imread("D:\python_code\Images\sou.png")
imgDst = cv2.imread("D:\python_code\Images\m3.png")







# L'algorithme de la détection par landmarks nécessitent des images en gray      
imgSrc_gray = cv2.cvtColor(imgSrc, cv2.COLOR_BGR2GRAY)
imgDst_gray = cv2.cvtColor(imgDst, cv2.COLOR_BGR2GRAY)
    
# Le pentagone dont la forme ressemple à ça 
"""
                    ______________
               /                   \
              \                    \
               \                   /
                \                 / 
                 \               / 
                  \             /
                   \ _________ /

"""
mask = np.zeros_like(imgSrc_gray)
new_face = np.zeros_like(imgDst) 

#initilisation de l'objet responsable de la détection du visage  
detector = dlib.get_frontal_face_detector()
#modèle pré-entrainé 
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#dans le cas d'images statique(!= real time) len(faces)=1 mais en 
# temps réel y en a plusieurs 
faces = detector(imgSrc_gray)
    
for face in faces:
  # on prélève du visage (qui est dans img_gray) les 68 points caractéristiques 
  landmarks = predictor(imgSrc_gray, face)
  landmarks_pointSrc = []
  for n in range(0,68):
      # les coordonnées  des landmarks dans le plan image  
      # pour visualiser le contenu de l'objet landmarks par exemple 
      # un print(dir(landmarks)) fait le boulot
      x = landmarks.part(n).x
      y = landmarks.part(n).y
      landmarks_pointSrc.append((x,y))
  
        
  pointsSrc = np.array(landmarks_pointSrc, np.int32) 
  #former un pentagone à partir des landmarks 
  hullSrc=cv2.convexHull(pointsSrc) 
  cv2.fillConvexPoly(mask,hullSrc,255)

  face_image_Src = cv2.bitwise_and(imgSrc, imgSrc, mask=mask)
  rect = cv2.boundingRect(hullSrc)
  (x,y,w,h) = rect

  subdiv = cv2.Subdiv2D(rect)
  subdiv.insert(landmarks_pointSrc)
    
  trianglesSrc = subdiv.getTriangleList()
  trianglesSrc = np.array(trianglesSrc, dtype=np.int32)
  indexes_triangles = []
  for t in trianglesSrc :
      pt1 = (t[0], t[1])
      pt2 = (t[2], t[3])
      pt3 = (t[4], t[5])
    
      
      # récupérer les landmarks   
      index_pt1 = np.where((pointsSrc == pt1).all(axis=1))
      # (array([51], dtype=int64),) on ne veut que 51 c'est pour ça 
      # la fonction extract_index en gros on veut pas des tabelaux numpy 
      index_pt1 = extract_index(index_pt1)
    
      index_pt2 = np.where((pointsSrc == pt2).all(axis=1))
      index_pt2 = extract_index(index_pt2)
    
      index_pt3 = np.where((pointsSrc == pt3).all(axis=1))
      index_pt3 = extract_index(index_pt3)
       
   
      if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
        triangle = [index_pt1, index_pt2, index_pt3]
        indexes_triangles.append(triangle)
    
     
         
            
    
facesDst = detector(imgDst_gray)

for face in facesDst:
    landmarks = predictor(imgDst_gray, face)
    landmarks_pointDst = []
    for n in range(0,68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        landmarks_pointDst.append((x,y))
    
    pointsDst = np.array(landmarks_pointDst, np.int32)
    hullDst = cv2.convexHull(pointsDst)
   
    
lines_space_mask = np.zeros_like(imgSrc_gray)
lines_space_new_face = np.zeros_like(imgDst)         
#triangulisation of the second face, from the first face 

for triangle_index in indexes_triangles:
    
    tr1_pt1 = landmarks_pointSrc[triangle_index[0]] # c'est pt1 in line 91
    tr1_pt2 = landmarks_pointSrc[triangle_index[1]] # c'est pt2   """   93
    tr1_pt3 = landmarks_pointSrc[triangle_index[2]] # c'est pt3
    triangleDst_i = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)
    
    #traitement des triangles un par 1 
    rectSrc = cv2.boundingRect(triangleDst_i)
    (x,y,w,h) = rectSrc
    cropped_triangleSrc = imgSrc[y:y+h, x:x+w]
    
    cropped_maskSrc = np.zeros((h,w), np.uint8)
    
    
    pointsSrc = np.array([[tr1_pt1[0]-x, tr1_pt1[1]-y],
                      [tr1_pt2[0]-x, tr1_pt2[1]-y],
                      [tr1_pt3[0]-x, tr1_pt3[1]-y]])
    

  
    lines_space = cv2.bitwise_and(cropped_triangleSrc,cropped_triangleSrc, mask=cropped_maskSrc)

    
    #triangulisation of the second face
    # On s'intéresse aux memes pooints caractéristiques 
    tr2_pt1 = landmarks_pointDst[triangle_index[0]]
    tr2_pt2 = landmarks_pointDst[triangle_index[1]]
    tr2_pt3 = landmarks_pointDst[triangle_index[2]]
    triangleDst = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)
    
    rectDst = cv2.boundingRect(triangleDst)
    (x,y,w,h) = rectDst

    
    cropped_maskDst = np.zeros((h,w), np.uint8)
    pointsDst = np.array([[tr2_pt1[0]-x, tr2_pt1[1]-y],
                      [tr2_pt2[0]-x, tr2_pt2[1]-y],
                      [tr2_pt3[0]-x, tr2_pt3[1]-y]])
    
    cv2.fillConvexPoly(cropped_maskDst, pointsDst, 255)
    
    
    #warp triangles 
    pointsSrc = np.float32(pointsSrc)
    pointsDst = np.float32(pointsDst)
    #effectuer la transformation affine 
    M = cv2.getAffineTransform(pointsSrc, pointsDst)
    # transformation du triangle source selon warpaffine 
    warped_triangle = cv2.warpAffine(cropped_triangleSrc, M, (w,h))
    #faire passer les triangles du visage1 sur le 2 en utilisantt un mask 
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_maskDst)

    # Reconstructing destination face
    new_faceRect = new_face[y: y + h, x: x + w]
    new_faceRect_gray = cv2.cvtColor(new_faceRect, cv2.COLOR_BGR2GRAY)
    _, mask_triangles_designed = cv2.threshold(new_faceRect_gray, 1, 255, cv2.THRESH_BINARY_INV)
    warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

    new_faceRect = cv2.add(new_faceRect, warped_triangle)
    new_face[y: y + h, x: x + w] = new_faceRect



          
        
          
#face swapped (mettre le premier visage dans le second)
img2_face_mask = np.zeros_like(imgDst_gray)
img2_head_mask = cv2.fillConvexPoly(img2_face_mask, hullDst, 255)
img2_face_mask = cv2.bitwise_not(img2_head_mask)


img2Noface = cv2.bitwise_and(imgDst, imgDst, mask=img2_face_mask)
imgOut = cv2.add(img2Noface, new_face)

(x, y, w, h) = cv2.boundingRect(hullDst)
center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

seamlessclone = cv2.seamlessClone(imgOut, imgDst, img2_head_mask, center_face2, cv2.NORMAL_CLONE)
cv2.imshow("seamlessclone", seamlessclone)
cv2.imshow("image1", imgSrc)
cv2.imshow("image2", imgDst)
cv2.waitKey(0)
cv2.destroyAllWindows()


print(emojize(":thumbs_up:"))












