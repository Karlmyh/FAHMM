import numpy as np
import math
from sklearn.linear_model import LassoCV

from sklearn import linear_model
from sklearn.linear_model import ElasticNetCV
import time

def logPlus(a,b):
    c=np.max([a,b])
    return(c + np.log(np.sum(np.exp(a-c)+np.exp(b-c))))

def logSumExp(a):
    if np.all(np.isinf(a)):
        return np.log(0)
    elif len(a)==0:
        return (np.log(0))
    else:
        b = np.max(a)
    return(b + np.log(np.sum(np.exp(a-b))))

def normal(x,y,beta,sigma):
    
    if(np.isnan(np.exp(-(y-np.matmul(x,beta))**2/(2*sigma))/np.sqrt(2*math.pi*sigma))):
        print("error")
        print("x",x)
        print("y",y)
        print("Beta",beta)
        print("sigma",sigma)
    return np.exp(-(y-np.matmul(x,beta))**2/(2*sigma))/np.sqrt(2*math.pi*sigma)


#compute the likelihood
#using log sum tricks
def normallog(x,y,beta,sigma):
    if(np.isnan(-(y-np.matmul(x,beta))**2/(2*sigma)-np.log(2*math.pi*sigma)/2)):
        print("error")
    return -(y-np.matmul(x,beta))**2/(2*sigma)-np.log(2*math.pi*sigma)/2


def adjustmentlog(series):
    series_sum=logSumExp(series)
    r=(np.log(0.1))/2-series_sum
    return series+r

#pi initial distribution N
#P transition matrix N*N
#X var p*T
#Y depvar T
#beta coef p*T*S
#sigma var of gaussian 
#########using log exp tricks
######### multiple input 
def forwardlog(pi,P,X,Y,Beta,Sigma):
    T=Y.shape[0]
    S=Y.shape[1]
    N=len(pi)
    alpha=np.ones(N*T).reshape([T,N])*np.log(0)
    #initialization
    #only alpha and normal likelihood are stored in log form
    for s in range(N):
        for t in range(S):
            alpha[0,s]=np.log(pi[s])+normallog(X[0,:],Y[0,t],Beta[s,:,t],Sigma[s,t])
    alpha[0,:]=adjustmentlog(alpha[0,:])

    
    #alpha=np.log(alpha)
    #iteration
    for i in range(1,T):
        for s1 in range(N):
            for s2 in range(N):
                for t in range(S):
                    increment=alpha[i-1,s2]+np.log(P[s2,s1])+normallog(X[i,:],Y[i,t],Beta[s1,:,t],Sigma[s1,t])
                    total=logSumExp(np.array([alpha[i,s1],increment]))
                    alpha[i,s1]=total
        alpha[i,:]=adjustmentlog(alpha[i,:])
    #alpha=np.apply_along_axis(adjustment,1,alpha)
    return np.exp(alpha)

######### multiple input 
def backwardlog(pi,P,X,Y,Beta,Sigma):
    T=Y.shape[0]
    S=Y.shape[1]
    N=len(pi)
    beta=np.ones(N*T).reshape([T,N])*np.log(0)
    
    #initialization
    for s in range(N):
        beta[T-1,s]=0
    
    #iteration
    for i in range(T-2,-1,-1):
        for s1 in range(N):
            for s2 in range(N):
                for t in range(S):
                    increment=np.log(P[s1,s2])+normallog(X[i+1,:],Y[i+1,t],Beta[s2,:,t],Sigma[s2,t])+beta[i+1,s2]
                    total=logSumExp(np.array([beta[i,s1],increment]))
                    beta[i,s1]=total
                
        beta[i,:]=adjustmentlog(beta[i,:])
    #beta=np.apply_along_axis(adjustment,1,beta)
    return np.exp(beta)


def norm(A,ord=1):
    if ord==1:
        return np.sum(np.abs(A))
    if ord==2:
        return np.sqrt(np.sum(np.abs(A)**2))
    

def fitted(xi,X,Beta):
    T=xi.shape[0]
    N=xi.shape[1]
    dim=Beta.shape[1]
    S=Beta.shape[2]
    coef=np.zeros([T,dim,S])
    for i in range(T):
        for ii in range(dim):
            for iii in range(S):
                for iiii in range(N):
                    coef[i,ii,iii]+=xi[i,iiii]*Beta[iiii,ii,iii]
    
    reg=np.zeros([T,S])
    for i in range(T):
        reg[i,:]=np.transpose(coef[i,:,:]).dot(X[i])
    return(reg)







######### multiple input version
#Y T*S


def baum_welchlog(X,Y,N,multiple,method,tol=1e-7,seed=666,printevery=1,nancheck=0,randomizer="diagprin"):
    #tolerance of convergence
    #cross validation fold
    numCV = 10
    #maximum iteration of linear regression before convergence
    numiter=15000
    #maximum iteration of EM
    iterations=500
    T=Y.shape[0]
    S=Y.shape[1]
    p=X.shape[1]
    Y=Y*multiple
    status="Normal"
    np.random.seed(seed)
    
    
            #P=1/N*np.ones([N,N])
            #P_randomizer=np.random.dirichlet(np.ones(N),size=N)/10
    # keep diagonal domination
    if randomizer=="diagprin":
        P_randomizer=np.random.rand(N*N).reshape([N,N])*2+np.diag(np.ones(N)*2)
    elif randomizer=="offdiagprin":
        P_randomizer=np.random.rand(N*N).reshape([N,N])*2-np.diag(np.ones(N)/N)
    elif randomizer=="none":
        P_randomizer=0
    
    #randomly initialize transition 
    P=1/N*np.ones([N,N])+P_randomizer
    P=P/np.sum(P,axis=1)
    #initialize beta as regression coef
    b=np.zeros([p,S])
    for i in range(S):
        if method=='elasticnet':
            model = ElasticNetCV(cv=numCV, random_state=0).fit(X, Y[:,i])
            b[:,i]=model.coef_
        elif method=='lasso':
            model= LassoCV(cv=numCV, random_state=0,max_iter=numiter).fit(X, Y[:,i])
            b[:,i]=model.coef_
        elif method=='scad':
            #multi-input need to be checked
            #model= pycasso.core.Solver(X,Y,penalty='scad')
            b=model.coef()['beta']
            b=b[-1,:]
        elif method=='simple':
            model=linear_model.LinearRegression().fit(X,Y[:,i])
            b[:,i]=model.coef_
    #model=regr.fit(X1,Y1)
    
    # beta N*p*S
    Beta=np.array([b for _ in range(N)])
    beta_randomizer=np.array([np.random.dirichlet(np.ones(S),size=p)*norm(Beta,ord=1) for _ in range(N)])
    #do not randomly adjust beta
    Beta=Beta+beta_randomizer

    #s S*1
    sigma=np.zeros(S)
    for i in range(S):
        sigma[i]=np.matmul(np.transpose(Y[:,i]-np.matmul(X,b[:,i])),Y[:,i]-np.matmul(X,b[:,i]))/T
    #Sigma N*S    
    Sigma=np.array([sigma for _ in range(N)])
    
    # save beta for termination
    preBeta=np.zeros([N,p,S])
    
    count=0
    criterionodd=0
    criterioneven=0
    
    time_s=time.time()
    for _ in range(iterations):
        
        count+=1
        
        #initial distribution is stationary
        w,v=np.linalg.eig(P)
        pi=abs(v[:,np.argmax(w)])
        if(nancheck):
                print("nancheck")
                print("Sigma",Sigma)
        
        alpha=forwardlog(pi,P,X,Y,Beta,Sigma)
        beta=backwardlog(pi,P,X,Y,Beta,Sigma)
            
        if(nancheck):
                print("nancheck")
                print("alpha",alpha)
                print("beta",beta)
                
        # updating tau the joint probability of adjacent latent states
        tau=np.zeros((T-1,N,N))
        
        for i in range(T-1):
            for s1 in range(N):
                for s2 in range(N):
                    normalloglike=0
                    for t in range(S):
                        normalloglike+=normallog(X[i+1,:],Y[i+1,t],Beta[s2,:,t],Sigma[s2,t])
                    tau[i,s1,s2]=np.log(alpha[i,s1])+np.log(P[s1,s2])+normalloglike+np.log(beta[i+1,s2])
                    if(np.isnan(tau[i,s1,s2])):
                        print("here","alpha",alpha[i,s1]*P[s1,s2],"normal",normal(X[i+1,:],Y[i+1],Beta[s2],Sigma[s2]),"beta",beta[i+1,s2])
        
        tau_sum=np.zeros(T-1)
        tau_sumfirst=np.zeros([T-1,N])
        for i in range(T-1):
                for s1 in range(N):
                    tau_sumfirst[i,s1]=logSumExp(tau[i,s1,:])
                tau_sum[i]=logSumExp(tau_sumfirst[i,:])
        
        
        # normalization
        for i in range(T-1):
            for s1 in range(N):
                for s2 in range(N):
                    tau[i,s1,s2]=tau[i,s1,s2]-tau_sum[i]
                    if(np.isnan(tau[i,s1,s2])):
                        print("here","alpha",alpha[i,s1]*P[s1,s2],"normal",normal(X[i+1,:],Y[i+1],Beta[s2],Sigma[s2]),"beta",beta[i+1,s2])
        #this is kesi!!!
        tau=np.exp(tau)
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
            if(nancheck):
                print("nancheck")
                print("Xi",xi[:,s])
            #probalistic separation
            #softly tends to 0 and 1
            #0.5 is conform to MLE
            
            X1=np.matmul(pow(Ws,0.5),X[:T-1,:])
            Y1=np.matmul(pow(Ws,0.5),Y[:T-1,:])
            if(nancheck):
                print("nancheck")
                print("X",X1[1:5],"Y",Y1[1:5])
            for t in range(S):
                if method=='elasticnet':
                    model = ElasticNetCV(cv=numCV, random_state=0).fit(X1, Y1)
                    Beta[s]=model.coef_
                elif method=='lasso':
                    #only lasso adjusted
                    model= LassoCV(cv=numCV, random_state=0,max_iter=numiter).fit(X1, Y1[:,t])
                    Beta[s,:,t]=model.coef_
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
            for t in range(S):
                Sigma[s,t]=np.matmul(np.transpose(Y1[:,t]-np.matmul(X1,Beta[s,:,t])),Y1[:,t]-np.matmul(X1,Beta[s,:,t]))/np.sum(xi[:,s])
            if(printevery):
                if (count%printevery==0):
                    print("Sigma",Sigma[s])
        #termination
        
        criterion=norm(Beta-preBeta,ord=2)/tol/norm(Beta,ord=2)
        
        #prevent cycle
        if (count%2==0):
            if (criterion==criterioneven):
                status="cycle"
                time_e=time.time()
                time_c=time_e-time_s
                print('status of %s ended at %d iteration, convergence: %.3f, time:%.3f s, and P now is ' %
                      ( status,count, criterion,time_c,))
                print(P)
                break
            criterioneven=criterion
        else:
            if (criterion==criterionodd):
                status="cycle"
                time_e=time.time()
                time_c=time_e-time_s
                print('status of %s ended at %d iteration, convergence: %.3f, time:%.3f s, and P now is ' %
                      ( status,count, criterion,time_c,))
                print(P)
                break
            criterionodd=criterion
        
        if(criterion<1):
            time_e=time.time()
            time_c=time_e-time_s
            print('ended at %d iteration, convergence: %.3f, time:%.3f s, and P now is ' %
                      ( count, criterion,time_c,))
            print(P)
            break
        else:
            time_e=time.time()
            time_c=time_e-time_s
            if(printevery):
                if (count%printevery==0):
                    print('%d iteration, convergence: %.3f, time:%.3f s, and P now is ' %
                      ( count, criterion,time_c,))
                    print(P)
                    
        #save result
        #print(preBeta[:,1:4])
        #print(Beta[:,1:4])
        preBeta=np.copy(Beta)
        
    

        
    le=np.log(0)
    for i in range(T-1):
        for s in range(N):
            for t in range(S):
                le=logSumExp(np.array([le,normallog(X[i,:],Y[i,t],Beta[s,:,t],Sigma[s,t])+np.log(xi[i,s])]))
        
    aic=-le*2+2*N
    return np.round(P,3),np.round(Beta,3),np.round(np.sqrt(Sigma),3),np.round(xi,3), aic
