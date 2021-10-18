import numpy as np
from baum_welchlog import *

# get fitted value
def fitted(xi,X,Beta,only=-1):
    ksi=np.copy(xi)
    if only==0:
        ksi[:,0]=1
        ksi[:,1]=0
    elif only==1:
        ksi[:,0]=0
        ksi[:,1]=1
    coef=ksi.dot(Beta)
    reg=coef*X[:-1,]
    return(np.sum(reg,axis=1))

# give simulation result
def evaluesim(P,Ptrue,Beta,Betatrue,Sigma,Sigmatrue,xi,xitrue,observe,reg):
    N=P.shape[0]
    #get codebook
    
    num=np.zeros(N)
    numtrue=np.zeros(N)
    for j in range(N):
        numtrue[j]=np.sum(xitrue==j)
    for i in range(N):
        num[i]=np.sum(np.argmax(xi,axis=1)==i)
    
    codebook=np.zeros(N)
    for i in range(N):
        codebook[i]=np.argmin(np.abs(numtrue-num[i])) 
    
    codebook=codebook.astype(int)
    T=xitrue.shape[0]
    MSE=np.sqrt(np.sum((reg[:-1]-fitted(xi,observe,Beta))**2)/T)
    #change states
    xitrue=xitrue[:-1]
    xi=np.argmax(xi,axis=1)
    for i in range(T-1):
        xi[i]=(codebook[xi[i]]==xitrue[i])

    states=np.sum(xi)/(T-1)
    #get P
    Ppermu=np.zeros([N,N])
    for i in range(N):
        Ppermu[codebook[i],:]=P[i,:]
    P=np.copy(Ppermu)
    for j in range(N):
        Ppermu[:,(codebook[j])]=P[:,j]
    print(Ppermu)
    transitionf=np.sum((Ppermu-Ptrue)**2)
    transitionmax=np.max(np.abs(Ppermu-Ptrue))
    
    #check Beta
    coef1=np.zeros(N+1)
    coef2=np.zeros(N+1)
    modelsize=np.array([np.sum(np.abs(Beta[0])>1e-8),np.sum(np.abs(Beta[1])>1e-8)])
    for i in range(N):
        coef2[i]=np.linalg.norm(Betatrue[:,codebook[i]]-Beta[i],ord=2)
        coef1[i]=np.linalg.norm(Betatrue[:,codebook[i]]-Beta[i],ord=1)
    
    coef1[N]=np.sum(coef1)
    coef2[N]=np.sqrt(np.sum(coef1**2))

    return np.concatenate([coef1,coef2,np.array([Sigma[0],Sigma[1],transitionf,transitionmax,modelsize[0],modelsize[1],MSE,states])])

# generate simulation data
def generatesim(dim,length=200,seed=666):
    # high dim two states case
    np.random.seed(seed)
    latent=np.array([0])
    # 3 latent states
    sigma=1

    #dim=100
    apart=50
    length=200
    identity=10
    sparse=7
    coef=np.zeros([dim,2])
    for i in range(2):
        coef[i*apart:(i*apart+sparse),i]=identity-3*i*identity
    covar=np.zeros([dim,dim])
    for i in range(dim):
        for j in range(dim):
            covar[i,j]=np.abs(i-j)
    covar=np.power(0.5,covar)
    observe=np.random.multivariate_normal(mean=np.zeros(dim),cov=covar,size=length)


    reg=np.array([np.matmul(observe[0,:],coef[:,0])]+np.array(np.random.normal(scale=sigma)))
    for s in range(length-1):
        #latent
        if(latent[-1]==0):
            latent=np.append(latent,np.random.choice([0,1],p=[0.9,0.1]))
        elif(latent[-1]==1):
            latent=np.append(latent,np.random.choice([0,1],p=[0.2,0.8]))
        else:
            print("error")

        #depvar
        if(latent[-1]==0):
            newdep=np.array([np.matmul(observe[s+1,:],coef[:,0])]+np.array(np.random.normal(scale=sigma)))
            reg=np.append(reg,newdep)
        elif(latent[-1]==1):
            newdep=np.array([np.matmul(observe[s+1,:],coef[:,1])]+np.array(np.random.normal(scale=sigma)))
            reg=np.append(reg,newdep)
        else:
            print("error")
    return observe,reg,coef,latent

# a whole simulation-train-evaluation process

def getsimresult(dim,method,seed1,seed2):
    observe,reg,coef,latent=generatesim(dim,seed1)
    Ptrue=np.array([[0.9,0.1],[0.2,0.8]])
    Betatrue=coef
    Sigmatrue=np.array([1,1])
    xitrue=latent
    P,Beta,Sigma,xi,aic,xilast = baum_welchlog(observe, reg,2,1,method,tol=1e-5,seed=seed2,nancheck=0,printevery=5,infocri="bic")

    return evaluesim(P,Ptrue,Beta,Betatrue,Sigma,Sigmatrue,xi,xitrue,observe,reg)


###### out of sample simulation 

# predict n step forward regimes
def nstepregime(P,xilast,n=1):
    regime=np.zeros(n).astype(np.int32)
    prob=xilast.dot(P)
    regime[0]=np.random.choice([0,1],p=prob)
    for i in range(1,n):
        regime[i]=np.random.choice([0,1],p=P[regime[i-1]])
    return regime

# predict n step forward value given regime
def nstepvalue(regime,Beta,feature):
    #regime n*2
    #Beta 2*p
    #feature n*p
    n=np.shape(regime)[0]
    xi=np.zeros([n,2])
    for i in range(n):
        xi[i,regime[i]]=1
    coef=xi.dot(Beta)
    pre=coef*feature
    result=np.sum(pre,axis=1)
    return result

# adjust permutation of two states
def transferregime(P,Beta,xilast):
    Pt=np.copy(P)
    Betat=np.copy(Beta)
    xilastt=np.copy(xilast)
    for i in range(2):
        for j in range(2):
            Pt[i,j]=P[1-i,1-j]
        Betat[i]=Beta[1-i]
        xilastt[i]=xilast[1-i]
    return Pt,Betat,xilastt


# main function for out of sample test
def outofsample(method,trainlength,ahead,observe,reg,latent,seed):
    T=reg.shape[0]
    mse=0
    regimenum=0
    count=0
    for i in range(trainlength,T-ahead+1):
        print(i)
        #from i th to i+ahead-1 th
        xitrue=latent[range(i,i+ahead)]
        regtrue=reg[range(i,i+ahead)]
        #train model on 0 to i-1 th
        P,Beta,Sigma,xi,aic,xilast=baum_welchlog(observe[range(i)], reg[range(i)],2,1,method,tol=1e-5,printevery=20,seed=seed)
        #switching
        if np.argmax(xi,axis=1)[0]==1:
            P,Beta,xilast=transferregime(P,Beta,xilast)
        #pre regime and value
        
        regime=nstepregime(P,xilast,n=ahead)
        
        value=nstepvalue(regime,Beta,observe[range(i,i+ahead)])
        mse+=np.sum((value-regtrue)**2)
        print("mse",mse)
        regimenum+=np.sum((regime-xitrue)**2)
        print("regimenum",regimenum)
        count+=ahead
        print("count",count)
        
    return np.sqrt(mse/count),regimenum/count
            
        