import numpy as np
import math
from sklearn.linear_model import LassoCV

from sklearn import linear_model
from sklearn.linear_model import ElasticNetCV
import time

#compute the likelihood
#using log sum tricks
def normal(x,y,beta,sigma):
    
    if(np.isnan(np.exp(-(y-np.matmul(x,beta))**2/(2*sigma))/np.sqrt(2*math.pi*sigma))):
        print("error")
        print("x",x)
        print("y",y)
        print("Beta",beta)
        print("sigma",sigma)
    return np.exp(-(y-np.matmul(x,beta))**2/(2*sigma))/np.sqrt(2*math.pi*sigma)


def adjustment(series):
    series_sum=np.sum(series)
    r=(math.log(0.1/series_sum,10)+math.log(1/series_sum,10))/2
    return series*pow(10,r)

#pi initial distribution N
#P transition matrix N*N
#X var p*T
#Y depvar T
#beta coef p*N
#sigma var of gaussian 
def forward(pi,P,X,Y,Beta,Sigma):
    T=len(Y)
    N=len(pi)
    alpha=np.zeros((T,N))
    
    #initialization
    for s in range(N):
        alpha[0,s]=pi[s]*normal(X[0,:],Y[0],Beta[s],Sigma[s])
    alpha[0,:]=adjustment(alpha[0,:])
    #alpha=np.log(alpha)
    #iteration
    for i in range(1,T):
        for s1 in range(N):
            for s2 in range(N):
                alpha[i,s1]+=alpha[i-1,s2]*P[s2,s1]*normal(X[i,:],Y[i],Beta[s1],Sigma[s1])
        alpha[i,:]=adjustment(alpha[i,:])
    
    #alpha=np.apply_along_axis(adjustment,1,alpha)
    return alpha

def backward(pi,P,X,Y,Beta,Sigma):
    T=len(Y)
    N=len(pi)
    beta=np.zeros((T,N))
    
    #initialization
    for s in range(N):
        beta[T-1,s]=1
    
    #iteration
    for i in range(T-2,-1,-1):
        for s1 in range(N):
            for s2 in range(N):
                beta[i,s1]+=P[s1,s2]*normal(X[i+1,:],Y[i+1],Beta[s2],Sigma[s2])*beta[i+1,s2]
                #if(i>T-5):
                    #print([i,s1,s2],"diff",Y[i+1]-np.matmul(X[i+1,:],Beta[s2]))
                    #print("normal",normal(X[i+1,:],Y[i+1],Beta[s2],Sigma[s2]))
            
        beta[i,:]=adjustment(beta[i,:])
    #beta=np.apply_along_axis(adjustment,1,beta)
    return beta




def baum_welch(X,Y,N,multiple,method,tol=1e-7,seed=666,printevery=1,nancheck=0):
    #tolerance of convergence
    #cross validation fold
    numCV = 10
    #maximum iteration before convergence
    numiter=15000
    iterations=1500
    T=len(Y)
    p=X.shape[1]
    Y=Y*multiple
    np.random.seed(seed)
    #P=1/N*np.ones([N,N])
    #P_randomizer=np.random.dirichlet(np.ones(N),size=N)/10
    P_randomizer=np.random.rand(N*N).reshape([N,N])*2
    beta_randomizer=np.random.dirichlet(np.ones(p),size=N)/10
    #randomly initialize transition 
    P=1/N*np.ones([N,N])+P_randomizer
    P=P/np.sum(P,axis=1)
    #initialize beta as regression coef
    if method=='elasticnet':
        model = ElasticNetCV(cv=numCV, random_state=0).fit(X, Y)
        b=model.coef_
    elif method=='lasso':
        model= LassoCV(cv=numCV, random_state=0,max_iter=numiter).fit(X, Y)
        b=model.coef_
    elif method=='scad':
        #model= pycasso.core.Solver(X,Y,penalty='scad')
        b=model.coef()['beta']
        b=b[-1,:]
    elif method=='simple':
        model=linear_model.LinearRegression().fit(X,Y)
        b=model.coef_
    #model=regr.fit(X1,Y1)
    
    Beta=[b for _ in range(N)]
    
    #do not randomly adjust beta
    Beta=Beta+beta_randomizer

    
    s=np.matmul(np.transpose(Y-np.matmul(X,b)),Y-np.matmul(X,b))/T
    Sigma=[s for _ in range(N)]
    
    preBeta=np.zeros([N,p])
    
    count=0
    time_s=time.time()
    for _ in range(iterations):
        #print(np.sum(np.abs(Beta[1]-Beta[0])))
        count+=1
        #initial distribution is stationary
        w,v=np.linalg.eig(P)
        pi=abs(v[:,np.argmax(w)])
        if(nancheck==1):
                print("nancheck")
                print("Sigma",Sigma)
        alpha=forward(pi,P,X,Y,Beta,Sigma)
        beta=backward(pi,P,X,Y,Beta,Sigma)
        
            
        if(nancheck==1):
                print("nancheck")
                print("alpha",alpha)
                print("beta",beta)
        # updating tau the joint probability of adjacent latent states
        tau=np.zeros((T-1,N,N))
        
        for i in range(T-1):
            for s1 in range(N):
                for s2 in range(N):
                    tau[i,s1,s2]=alpha[i,s1]*P[s1,s2]*normal(X[i+1,:],Y[i+1],Beta[s2],Sigma[s2])*beta[i+1,s2]
                    if(np.isnan(tau[i,s1,s2])):
                        print("here","alpha",alpha[i,s1]*P[s1,s2],"normal",normal(X[i+1,:],Y[i+1],Beta[s2],Sigma[s2]),"beta",beta[i+1,s2])
                    
        
        tau_sum=np.sum(np.sum(tau,axis=1),axis=1)
        
        # normalization
        for i in range(T-1):
            for s1 in range(N):
                for s2 in range(N):
                    tau[i,s1,s2]=tau[i,s1,s2]/tau_sum[i]
                    if(np.isnan(tau[i,s1,s2])):
                        print("here","alpha",alpha[i,s1]*P[s1,s2],"normal",normal(X[i+1,:],Y[i+1],Beta[s2],Sigma[s2]),"beta",beta[i+1,s2])
        #this is kesi!!!
        
        xi=np.sum(tau,axis=2)
        #print(tau)
        #averaging over time
        tau2=np.sum(tau,axis=0)
        #averaging over time
        xi_sum=np.sum(xi,axis=0)
        
        #conditional probability
        for s1 in range(N):
            for s2 in range(N):
                P[s1,s2]=tau2[s1,s2]/xi_sum[s1]
                #print([tau2[s1,s2],xi_sum[s1]])
        
        #updating beta which govern the emmission process, check inside 
        for s in range(N):
            
            Ws=np.diag(xi[:,s])
            if(nancheck==1):
                print("nancheck")
                print("Xi",xi[:,s])
            #probalistic separation
            #softly tends to 0 and 1
            #0.5 is conform to L2 norm and probalistic framework
            #
            X1=np.matmul(pow(Ws,0.5),X[:T-1,:])
            Y1=np.matmul(pow(Ws,0.5),Y[:T-1])
            if(nancheck==1):
                print("nancheck")
                print("X",X1[1:5],"Y",Y1[1:5])
            if method=='elasticnet':
                model = ElasticNetCV(cv=numCV, random_state=0).fit(X1, Y1)
                Beta[s]=model.coef_
            elif method=='lasso':
                model= LassoCV(cv=numCV, random_state=0,max_iter=numiter).fit(X1, Y1)
                Beta[s]=model.coef_
            elif method=='scad':
                #model= pycasso.core.Solver(X1,Y1,penalty='scad')
                Beta[s]=model.coef()['beta'][-1,:]
            elif method=='simple':
                model==linear_model.LinearRegression().fit(X1,Y1)
                Beta[s]=model.coef_
                
            if method=='simple':
                n_s=0
            else:
                n_s=sum(Beta[s]>0)
            #Sigma[s]=np.matmul(np.transpose(Y1-np.matmul(X1,Beta[s])),Y1-np.matmul(X1,Beta[s]))/(np.sum(xi[:,s])-n_s)
            Sigma[s]=np.matmul(np.transpose(Y1-np.matmul(X1,Beta[s])),Y1-np.matmul(X1,Beta[s]))/np.sum(xi[:,s])
            if (count%printevery==0):
                print("Sigma",Sigma[s])
        #termination
        criterion=np.linalg.norm(Beta-preBeta,ord=2)/tol/np.linalg.norm(Beta)
        if(criterion<1):break
        else:
            time_e=time.time()
            time_c=time_e-time_s
            if (count%printevery==0):
                print('%d iteration, convergence: %.3f, time:%.3f s, and P now is ' %
                  ( count, criterion,time_c,))
                print(P)
        #save result
        preBeta=np.copy(Beta)
        
        
    le,llk=0,0
    for i in range(T-1):
        for s in range(N):
            le+=normal(X[i,:],Y[i],Beta[s],Sigma[s])*xi[i,s]
        llk+=np.log(le)
    aic=-llk*2+2*N
    return np.round(P,3),np.round(Beta,3),np.round(np.sqrt(Sigma),3),np.round(xi,3), aic

