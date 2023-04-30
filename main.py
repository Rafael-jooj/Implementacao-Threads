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
        print('processando mascara de sobel 1')
    
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
        print('processando mascara de sobel 2')
    
    image_result = Image.fromarray(newImage.astype(np.uint8))
    image_result.show()



def dilatacao(image):
    mask = np.array([[0,1,0], [1,1,1], [0,1,0]], dtype = int)

    linha = (image.shape[0])
    coluna = (image.shape[1])

    newImage = np.zeros((linha, coluna))

    for i in range(1, linha-1):
        for j in range(1, coluna-1):
            maior = [image[i-1][j-1] * mask[0][0], image[i-1][j] * mask[0][1], image[i-1][j+1] * mask[0][2], image[i][j-1] * mask[1][0], image[i][j] * mask[1][1], image[i][j+1] * mask[1][2], image[i+1][j-1] * mask[2][0], image[i+1][j] * mask[2][1], image[i+1][j+1] * mask[2][2]]
            newImage[i][j] = max(maior, key=int)
        print('processando dilatação')

    img = Image.fromarray(newImage.astype(np.uint8))
    img.show()



def erosao(image):
    mask2 = np.array([[0,1,0], [1,1,1], [0,1,0]], dtype = int)

    linha2 = (image.shape[0])
    coluna2 = (image.shape[1])

    newImage2 = np.zeros((linha2, coluna2))

    for i in range(1, linha2-1):
        for j in range(1, coluna2-1):
            menor = [image[i-1][j] * mask2[0][1], image[i][j-1] * mask2[1][0], image[i][j] * mask2[1][1], image[i][j+1] * mask2[1][2], image[i+1][j] * mask2[2][1]]
            newImage2[i][j] = min(menor, key=int)
        print('processando erosao')

    img = Image.fromarray(newImage2.astype(np.uint8))
    img.show()



inicio = time.time()

# filtro_sobel_mask1(imgArray) #teste sem thread
main1 = threading.Thread(target=filtro_sobel_mask1, args=(imgArray,))
main1.start()

# filtro_sobel_mask2(imgArray) #teste sem thread
main2 = threading.Thread(target=filtro_sobel_mask2, args=(imgArray,))
main2.start()

# erosao(imgArray) #teste sem thread
main3 = threading.Thread(target=erosao, args=(imgArray,))
main3.start()

# dilatacao(imgArray) #teste sem thread
main4 = threading.Thread(target=dilatacao, args=(imgArray,))
main4.start()

main1.join()
main2.join()
main3.join()
main4.join()

fim = time.time()

print(fim - inicio)

#TESTES EFETUADOS NA MINHA MÁQUINA
#Tempo de execução sem utilização de Threads = 18.6 segundos
#Tempo de execução com a utilização de Threads = 9.13 segundos
