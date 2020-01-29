import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.linear_model
data = np.loadtxt("notas_andes.dat", skiprows=1)
Y = data[:,4]
X = data[:,:4]
regresion = sklearn.linear_model.LinearRegression()
Tr=1000
Betas=np.zeros((Tr,4))
Beta0=np.zeros(Tr)

for i in range(Tr):
    l=np.random.randint(0,69,size=69)
    YT=Y[l]
    XT=X[l,:]
    regresion.fit(XT, YT)
    Betas[i,:]=regresion.coef_
    Beta0[i]=regresion.intercept_ 
plt.figure(figsize=(12,12))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.hist(Betas[:,i])
    plt.title('Beta'+str (i+1) +'=%4.3f' %np.mean(Betas[:,i])+'$\pm$ %4.3f' %np.std(Betas[:,i]))
plt.savefig('bootstrap.png')