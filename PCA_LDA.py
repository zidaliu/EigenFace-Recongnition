import os
import numpy as np
import scipy.linalg as linalg
import cv2
import operator
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import pyplot

def ComputeNorm(x):
    # function r=ComputeNorm(x)
    # computes vector norms of x
    # x: d x m matrix, each column a vector
    # r: 1 x m matrix, each the corresponding norm (L2)

    [row, col] = x.shape
    r = np.zeros((1,col))

    for i in range(col):
        r[0,i] = linalg.norm(x[:,i])
    return r

def read_faces(directory):
    # function faces = read_faces(directory)
    # Browse the directory, read image files and store faces in a matrix
    # faces: face matrix in which each colummn is a colummn vector for 1 face image
    # idLabels: corresponding ids for face matrix

    A = []  # A will store list of image vectors
    Label = [] # Label will store list of identity label
 
    # browsing the directory
    for f in os.listdir(directory):
        if not f[-3:] =='bmp':
            continue
        infile = os.path.join(directory, f)
        im = cv2.imread(infile, 0)
        # turn an array into vector
        im_vec = np.reshape(im, -1)
        A.append(im_vec)
        name = f.split('_')[0][-1]
        Label.append(int(name))

    faces = np.array(A, dtype=np.float32)
    faces = faces.T
    idLabel = np.array(Label)
    return faces,idLabel

def myPCA(A):
    # function [W,LL,m]=mypca(A)
    # computes PCA of matrix A
    # A: D by N data matrix. Each column is a random vector
    # W: D by K matrix whose columns are the principal components in decreasing order
    # LL: eigenvalues
    # m: mean of columns of A
    # Note: "lambda" is a Python reserved word
    # compute mean, and subtract mean from every column

    [r,c] = A.shape #读取行列的维度
    m = np.mean(A,1) #列向量的平均值
    A = A - np.tile(m, (c,1)).T #使其平均值等于0
    B = np.dot(A.T, A) 
    [d,v] = linalg.eig(B) #特征向量的分解

    # sort d in descending order
    order_index = np.argsort(d)
    order_index =  order_index[::-1]
    # print(order_index)
    d = d[order_index]
    v = v[:, order_index]

    # compute eigenvectors of scatter matrix
    W = np.dot(A,v)
    Wnorm = ComputeNorm(W)

    W1 = np.tile(Wnorm, (r, 1))
    W2 = W / W1
 
    LL = d[0:-1]

    W = W2[:,0:-1]      #omit last column, which is the nullspace
    
    return W, LL, m

def myLDA(A,Labels):
    # function [W,m]=myLDA(A,Label)
    # computes LDA of matrix A
    # A: D by N data matrix. Each column is a random vector
    # W: D by K matrix whose columns are the principal components in decreasing order
    # m: mean of each projection


    classLabels = np.unique(Labels)
    classNum = len(classLabels)
    dim,datanum = A.shape
    totalMean = np.mean(A,1)
    partition = [np.where(Labels==label)[0] for label in classLabels]
    classMean = [(np.mean(A[:,idx],1),len(idx)) for idx in partition]

    #compute the within-class scatter matrix
    W = np.zeros((dim,dim))
    for idx in partition:
        W += np.cov(A[:,idx],rowvar=1)*len(idx)

    #compute the between-class scatter matrix
    B = np.zeros((dim,dim))
    for mu,class_size in classMean:
        offset = mu - totalMean
        B += np.outer(offset,offset)*class_size

    #solve the generalized eigenvalue problem for discriminant directions
    ew, ev = linalg.eig(B,W)
    sorted_pairs = sorted(enumerate(ew), key=operator.itemgetter(1), reverse=True)
    selected_ind = [ind for ind,val in sorted_pairs[:classNum-1]]
    LDAW = ev[:,selected_ind]
    Centers = [np.dot(mu,LDAW) for mu,class_size in classMean]
    Centers = np.array(Centers).T
    return LDAW, Centers, classLabels

def float2uint8(arr):
    mmin = arr.min()
    mmax = arr.max()
    arr = (arr-mmin)/(mmax-mmin)*255
    arr = np.uint8(arr)
    return arr




#############################################################################################################
# mycode-PCA-training
K=30 #设置K值
K1=90
[faces,idLabel] = read_faces('face/train')
[W,LL,m] = myPCA(faces)
We = W[:, : K]
We_1 = W[:, : K1]
for x in range(0,120):
    faces[:,x] = faces[:,x] - m
Y = np.dot(We.T,faces)
Y_1 = np.dot(We_1.T,faces)


#求training data的Z平均值，分成10类

classLabels = np.unique(idLabel)

classNum = len(classLabels)
dim,datanum = Y.shape
totalMean = np.mean(Y,1)
partition = [np.where(idLabel==label)[0] for label in classLabels]
classMean = [(np.mean(Y[:,idx],1)) for idx in partition]
classMean = np.array(classMean)
classMean = classMean.T


#mycode-PCA-testing

[faces1,idLabel1] = read_faces('face/test')
[W1,LL1,m1] = myPCA(faces1)
for x in range(0,120):
    faces1[:,x] = faces1[:,x] - m
Y1 = np.dot(We.T,faces1)
Y2 = np.dot(We_1.T,faces1)

#confusion matrix_PCA
[Row,Col] = Y1.shape
Con_Matrix = np.zeros((10,10))

for a in range(0,Col):
    temp = np.linalg.norm(Y1[:,a] - classMean[:,0]) #记录最小的值，初始一个很大的值
    save1 = 0 #存储正确的类号
    save2 = 0 #存储分类的类号
    for b in range(0,10):
        dis = np.linalg.norm(Y1[:,a] - classMean[:,b])

        if dis < temp:
            temp = dis
            save2 = b
            save1 = int(a/12)  

    Con_Matrix[save1][save2] = Con_Matrix[save1][save2]+1
print("PCA_Con_Matrix: ")
print(Con_Matrix)

#计算测试集在运行时的准确率
sum = 0
for i in range(0,10):
    sum += Con_Matrix[i][i]
print("k=%d时"%K)
print("PCA分类的准确性为: %f"%(sum/120))
print("PCA分类的错误率为: %f"%(1-(sum/120)))


# 可视化PCA_mean,以及前8个特征值
aver = m.reshape(160,140)
aver = float2uint8(aver)
Imk = Image.fromarray(aver)
fig1 = pyplot.figure(1)  
ave_r = fig1.add_subplot(339)
plt.title('Mean')
ave_r.imshow(Imk,"gray")

for x in range(2,10):
    eign = We[:,x-2]
    eign = eign.reshape(160,140)
    eign = float2uint8(eign)#转化到255
    Imk1 = Image.fromarray(eign)
    eignface = fig1.add_subplot(3,3,x-1)
    plt.title('K = %d'%(x-1))
    eignface.imshow(Imk1,"gray")



# mycode-LDA-training
[faces,idLabel] = read_faces('face/train')

[LDAW,Centers,classLabels] = myLDA(Y_1,idLabel)
Y_LDA = np.dot(LDAW.T,Y_1)
[Row,Col1] = Y_LDA.shape
Con_Matrix_LDA = np.zeros((10,10))

#mycode-LDA-testing
Y1_LDA = np.dot(LDAW.T,Y2)
#confusion matrix_LDA
for o in range(0,Col1):
    temp = 100000000000 #记录最小的值，初始一个很大的值
    save1 = 0 #存储正确的类号
    save2 = 0 #存储分类的类号
    for b in range(0,10):
        dis = np.linalg.norm(Y1_LDA[:,o] - Centers[:,b])
        if dis < temp:
            temp = dis
            save2 = b
            save1 = int(o/12)  
    Con_Matrix_LDA[save1][save2] = Con_Matrix_LDA[save1][save2]+1
    

print("LDA_Con_Matrix: ")
print(Con_Matrix_LDA)

sum = 0
for i in range(0,10):
    sum += Con_Matrix_LDA[i][i]
print("k=%d时"%K1)
print("LDA分类的准确性为: %f"%(sum/120))
print("LDA分类的错误率为: %f"%(1-(sum/120)))

#重建LDA~PCA~可视化图
Cp = np.dot(LDAW,Centers)
Cr = np.dot(We_1,Cp)
for u in range(0,10):
    Cr[:,u] = Cr[:,u] + m

fig2 = pyplot.figure(2) 

for p in range(0,10):
    eign = Cr[:,p]
    eign = eign.reshape(160,140)
    eign = float2uint8(eign)#转化到255
    Imk1 = Image.fromarray(eign)
    eignface = fig2.add_subplot(3,4,p+1)
    plt.title('Center =  %d'%(p+1))
    eignface.imshow(Imk1,"gray")



#计算fusion
classMean = classMean * 0.5
Centers = Centers * 0.5
fusion = np.vstack((classMean,Centers))#training data的标准
Y1 = Y1 * 0.5 #0.5Ye
Y1_LDA = Y1_LDA * 0.5 #0.5Yf
testing_data = np.vstack((Y1,Y1_LDA))#测试数据

#confusion matrix_fusion
Con_Matrix_Fusion = np.zeros((10,10))
for number in range(0,Col1):
    temp = 100000000000 #记录最小的值，初始一个很大的值
    save1 = 0 #存储正确的类号
    save2 = 0 #存储分类的类号
    for b in range(0,10):
        dis = np.linalg.norm(testing_data[:,number] - fusion[:,b])
        if dis < temp:
            temp = dis
            save2 = b
            save1 = int(number/12)  
    Con_Matrix_Fusion[save1][save2] = Con_Matrix_Fusion[save1][save2]+1
    
sum = 0
for i in range(0,10):
    sum += Con_Matrix_Fusion[i][i]

print("Fusion_Con_Matrix: ")
print(Con_Matrix_Fusion)
print("k_PCA=%d,K_LDA=%d"%(K,K1))
print("Fusion分类的准确性为: %f"%(sum/120))
print("Fusion分类的错误率为: %f"%(1-(sum/120)))


pyplot.show()







