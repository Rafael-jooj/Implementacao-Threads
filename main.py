import numpy as np
from PIL import Image
import threading
import time

# Implementação das mascaras de sobel em imagens em escala de cinza utilizando Threads para realizar as duas aplicações ao mesmo tempo

img = Image.open('processador.jpg')
img = img.convert('L')
imgArray = np.array(img)

def filtro_sobel_mask1(image):
    rows, cols = image.shape
    newImage = np.zeros((rows, cols), dtype=int)

    masksobel1 = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype= int) 

    for row in range(1, rows -1):
        for col in range(1, cols -1):
            result = image[row-1, col-1] * masksobel1[0,0] + image[row-1, col] * masksobel1[0,1] + image[row - 1, col+1] * masksobel1[0,2] + image[row, col-1] * masksobel1[1,0] + image[row, col] * masksobel1[1,1] + image[row, col+1] * masksobel1[1,2] + image[row+1, col-1] * masksobel1[2,0] + image[row+1, col] * masksobel1[2,1] + image[row+1, col+1] * masksobel1[2,2]
            if(result < 0):
                newImage[row, col] = 0
            else:
                newImage[row, col] = result
        print('processando mascara 1')
    
    image_result = Image.fromarray(newImage.astype(np.uint8))
    image_result.show()



def filtro_sobel_mask2(image):
    rows, cols = image.shape
    newImage = np.zeros((rows, cols), dtype=int)

    masksobel2 = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype= int)

    for row in range(1, rows -1):
        for col in range(1, cols -1):
            result = image[row-1, col-1] * masksobel2[0, 0] + image[row-1, col] * masksobel2[0, 1]+ image[row - 1, col+1] * masksobel2[0,2] + image[row, col-1] * masksobel2[1,0] + image[row, col] * masksobel2[1,1] + image[row, col+1] * masksobel2[1,2] + image[row+1, col-1] * masksobel2[2,0] + image[row+1, col] * masksobel2[2,1] + image[row+1, col+1] * masksobel2[2,2] 
            if(result < 0):
                newImage[row, col] = 0
            else:
                newImage[row, col] = result
        print('processando mascara 2')
    
    image_result = Image.fromarray(newImage.astype(np.uint8))
    image_result.show()


inicio = time.time()
# filtro_sobel_mask1(imgArray)
main1 = threading.Thread(target=filtro_sobel_mask1, args=(imgArray,))
main1.start()

# filtro_sobel_mask2(imgArray)
main2 = threading.Thread(target=filtro_sobel_mask2, args=(imgArray,))
main2.start()

main1.join()
main2.join()

fim = time.time()

print(fim - inicio)

#TESTES EFETUADOS NA MINHA MÁQUINA
#Tempo de execução sem utilização de Threads = 9.28 segundos
#Tempo de execução com a utilização de Threads = 5.97 segundos
