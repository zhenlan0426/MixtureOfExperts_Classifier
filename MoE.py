#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 10:42:51 2017

@author: will
"""

import numpy as np

def stable_softmax(X):
    temp = np.exp(X + np.mean(X,1,keepdims=True))
    return temp/np.sum(temp,1,keepdims=True)

def y2long(Y,k):
    n = Y.shape[0]
    Y_long = np.zeros((n,k))
    Y_long[range(n),Y] = 1
    return Y_long

def p2logit(p,k):
    temp = np.zeros(k)
    temp[:k-1]=np.log(p[:k-1]/p[k-1])
    return temp

class MixtureOfExperts_Classifier():
    # implement MLAPP,11.2.4 Mixtures of experts
    # both P(Y|X,Z) and P(Z|X) is modeled via logistic regressions
    
    def __init__(self,K1,K2,d,decay=None):
        # decay is the decay rate used in rmsprop
        self.K1 = K1 # for Y
        self.K2 = K2 # for Z, hyper-parameter
        self.d = d
        self.beta = np.random.randn(d,K1,K2)/d 
        self.theta = np.random.randn(d,K2)/d
        self.decay = decay
        if decay is not None:
            self.betaSq = np.zeros_like(self.beta)
            self.thetaSq = np.zeros_like(self.theta)
    
    def infer_Z_given_X(self,X):
        return stable_softmax(np.dot(X,self.theta))
        
    def infer_Y_given_XZ(self,X):
        # compute prob of Y|X,Z for all possible Y,Z
        # return array of shape (N,K1,K2)
        return stable_softmax(np.einsum('nd,dpq->npq', X, self.beta))
    
    def infer_Z_given_XY(self,X,Y,returnAll=False):
        # Y take val in {0,1,...K1-1}
        n = X.shape[0]
        P_Z_given_X = self.infer_Z_given_X(X)
        P_Y_given_XZ = self.infer_Y_given_XZ(X)
        Like_Y_given_XZ = P_Y_given_XZ[range(n),Y,:]
        temp = P_Z_given_X * Like_Y_given_XZ
        if returnAll:
            return temp/np.sum(temp,1,keepdims=True),P_Z_given_X,P_Y_given_XZ,Like_Y_given_XZ
        else:
            return temp/np.sum(temp,1,keepdims=True)
    
    def update(self,X,Y,learnR):
        # do one step update
        Z_givenXY, Z_givenX, Y_givenXZ,Like_Y_givenXZ = self.infer_Z_given_XY(X,Y,returnAll=True)
        
        n = X.shape[0]
        if self.decay is None: # SGD
            self.theta += learnR * np.dot(X.T,Z_givenXY-Z_givenX)/n # update theta
            Y_long = y2long(Y,self.K1) # update beta
            self.beta += learnR * np.einsum('nd,npq->dpq',X,\
                                            np.einsum('nq,npq->npq',Z_givenXY,np.expand_dims(Y_long,-1) - Y_givenXZ))/n
        
        else: # rmsprop
            G_theta = np.dot(X.T,Z_givenXY-Z_givenX)/n
            Y_long = y2long(Y,self.K1)
            G_beta = np.einsum('nd,npq->dpq',X,\
                                            np.einsum('nq,npq->npq',Z_givenXY,np.expand_dims(Y_long,-1) - Y_givenXZ))/n
            
            self.thetaSq = self.decay * self.thetaSq + (1-self.decay) * G_theta**2
            self.betaSq = self.decay * self.betaSq + (1-self.decay) * G_beta**2
            
            self.theta += learnR * G_theta / (self.thetaSq + 1e-6)
            self.beta += learnR * G_beta / (self.betaSq + 1e-6)
        
                                        
    def fit(self,learnR,iterN,batchSize,dataTrain,dataTest=None,score='acc'):
        # dataTest should be a tuple of (X,Y) for monitoring
        # dataTrain should be a tuple of (X,Y) for training
        # score is either accuracy or mle
        X_train, Y_train = dataTrain
        Y_train_long = y2long(Y_train,self.K1)
        if dataTest is not None:
            X_test, Y_test = dataTest
            Y_test_long = y2long(Y_test,self.K1)
            
        n = X_train.shape[0]
        batchN = int(n/batchSize)
        for i in range(iterN):
            index = np.random.permutation(n)
            for j in range(batchN):
                sub_index = index[j*batchSize:(j+1)*batchSize]
                self.update(X_train[sub_index],Y_train[sub_index],learnR)
            if dataTest is not None:
                scoreFun = lambda y,yhat: np.mean(np.argmax(y,1)==np.argmax(yhat,1)) \
                            if score == 'acc' else lambda y,yhat: np.mean(y*np.log(yhat)) 
                print 'iteration {}, train {}, test {}'\
                    .format(i,scoreFun(Y_train_long,self.predict_Y_given_X(X_train)), \
                              scoreFun(Y_test_long,self.predict_Y_given_X(X_test)))

    def predict_Y_given_X(self,X):
        # return P(Y|X) of shape (N,K1)
        P_Z_given_X = self.infer_Z_given_X(X)
        P_Y_given_XZ = self.infer_Y_given_XZ(X)
        return np.einsum('nq,npq->np',P_Z_given_X,P_Y_given_XZ)


''' test case
import matplotlib.pyplot as plt

n = 10000
d = 10
k1 = 4
k2 = 10
theta = np.random.randn(d,k2)
beta = np.random.randn(d,k1,k2)
def dataGen(n,d,theta,beta,noise=0.1):
    X = np.random.randn(n,d)
    Z = np.argmax(np.dot(X,theta) + np.random.randn(n,k2) * noise,1)
    Y = np.einsum('nd,dpq->npq',X,beta) + np.random.randn(n,k1,k2) * noise
    Y = np.argmax(Y[range(n),:,Z],1)
    return X,Y
dataTrain = dataGen(n,d,theta,beta,noise=0.1)
dataTest = dataGen(n,d,theta,beta,noise=0.1)
model = MixtureOfExperts_Classifier(k1,k2,d)
model.fit(1e-2,100,100,dataTrain,dataTest)

model = MixtureOfExperts_Classifier(k1,k2,d,0.9)
model.fit(1e-4,100,100,dataTrain,dataTest)

z = model.infer_Z_given_X(dataTrain[0])
plt.hist(z.flatten())

y = model.infer_Y_given_XZ(dataTrain[0])
np.std(y,2).max()
np.std(y,2).min()

'''


class MixtureOfExperts_Classifier2():
    # enable generic estimator in the place of P(Z|X) and P(Y|Z,X)
    # P(Y|Z,X) should be a list of lenth K2, where each element an estimator
    # estimator needs to support 2 API, 1) predict_proba(X) 2) update(gradient)
                        
    def __init__(self,K1,K2,d,estZ,estY):
        # decay is the decay rate used in rmsprop
        self.K1 = K1 # for Y
        self.K2 = K2 # for Z, hyper-parameter
        self.d = d
        self.estZ = estZ
        self.estY = estY

            
    def infer_Z_given_X(self,X,sub_index=None):
        return self.estZ.predict_proba(X,sub_index) if isinstance(self.estZ,GBM_KClass_MoE) \
                                                        else self.estZ.predict_proba(X)
        
    def infer_Y_given_XZ(self,X,sub_index=None):
        # compute prob of Y|X,Z for all possible Y,Z
        # return array of shape (N,K1,K2)
        return np.stack([est.predict_proba(X,sub_index) if isinstance(est,GBM_KClass_MoE) \
                                                        else est.predict_proba(X) for est in self.estY],-1)
    
    def infer_Z_given_XY(self,X,Y,sub_index,returnAll=False):
        # Y take val in {0,1,...K1-1}
        n = X.shape[0]
        P_Z_given_X = self.infer_Z_given_X(X,sub_index)
        P_Y_given_XZ = self.infer_Y_given_XZ(X,sub_index)
        Like_Y_given_XZ = P_Y_given_XZ[range(n),Y,:]
        temp = P_Z_given_X * Like_Y_given_XZ
        if returnAll:
            return temp/np.sum(temp,1,keepdims=True),P_Z_given_X,P_Y_given_XZ,Like_Y_given_XZ
        else:
            return temp/np.sum(temp,1,keepdims=True)
            
    def inference_grad(self,X,Y,sub_index):
        # calculate the gradient wrt pre-softmax layer for P(Z|X) and P(Y|X,Z)
        Y_long = y2long(Y,self.K1)
        Z_givenXY, Z_givenX, Y_givenXZ,Like_Y_givenXZ = self.infer_Z_given_XY(X,Y,sub_index,returnAll=True)
        return (Z_givenXY-Z_givenX), np.einsum('nq,npq->npq',Z_givenXY,np.expand_dims(Y_long,-1) - Y_givenXZ)
    
    def inference_grad_Z(self,X,Like_Y_given_XZ,sub_index):
        P_Z_given_X = self.infer_Z_given_X(X,sub_index)
        temp = P_Z_given_X * Like_Y_given_XZ
        return temp/np.sum(temp,1,keepdims=True) - P_Z_given_X
                                        
    def fit(self,iterN,batchSize,dataTrain,dataTest=None,score='acc',fixedY=False):
        # dataTest should be a tuple of (X,Y) for monitoring
        # dataTrain should be a tuple of (X,Y) for training
        # score is either accuracy or mle
        X_train, Y_train = dataTrain
        Y_train_long = y2long(Y_train,self.K1)
        if dataTest is not None:
            X_test, Y_test = dataTest
            Y_test_long = y2long(Y_test,self.K1)
            
        n = X_train.shape[0]
        for est_y in self.estY:
            if isinstance(est_y,GBM_KClass_MoE):
                est_y.cache = np.zeros((n,self.K1)) + est_y.baseline
                

        if isinstance(self.estZ,GBM_KClass_MoE):
            self.estZ.cache = np.zeros((n,self.K2)) + self.estZ.baseline    
                
        if dataTest is not None:
            scoreFun = lambda y,yhat: np.mean(np.argmax(y,1)==np.argmax(yhat,1)) \
                            if score == 'acc' else lambda y,yhat: np.mean(y*np.log(yhat)) 
        
        if fixedY:
            P_Y_given_XZ = self.infer_Y_given_XZ(X_train)
            Like_Y_given_XZ = P_Y_given_XZ[range(n),Y_train,:]
            if dataTest is not None:
                P_Y_given_XZ_test = self.infer_Y_given_XZ(X_test)
            
        batchN = int(n/batchSize)
        for i in range(iterN):
            index = np.random.permutation(n)
            for j in range(batchN):
                sub_index = index[j*batchSize:(j+1)*batchSize]
                non_index = index[range(j*batchSize)+range((j+1)*batchSize,n)]
                if fixedY:
                    grad_Z = self.inference_grad_Z(X_train[sub_index],Like_Y_given_XZ[sub_index],sub_index)
                else:
                    grad_Z, grad_Y = self.inference_grad(X_train[sub_index],Y_train[sub_index],sub_index)
                    # update P(Y|Z,X)
                    for k,est_y in enumerate(self.estY):
                        if isinstance(est_y,GBM_KClass_MoE):
                            est_y.update(X_train,grad_Y[:,:,k],sub_index,non_index)
                        else:
                            est_y.update(X_train[sub_index],grad_Y[:,:,k])
                    
                # update P(Z|X):

                if isinstance(self.estZ,GBM_KClass_MoE):
                    self.estZ.update(X_train,grad_Z,sub_index,non_index)
                else:
                    self.estZ.update(X_train[sub_index],grad_Z)
            
            if fixedY:
                if dataTest is not None: # monitor
                    print 'iteration {}, train {}, test {}'\
                        .format(i,scoreFun(Y_train_long,self.predict_Y_given_X_FixedY(X_train,P_Y_given_XZ)), \
                                  scoreFun(Y_test_long,self.predict_Y_given_X_FixedY(X_test,P_Y_given_XZ_test)))
            else:
                if dataTest is not None: # monitor
                    print 'iteration {}, train {}, test {}'\
                        .format(i,scoreFun(Y_train_long,self.predict_Y_given_X(X_train)), \
                                  scoreFun(Y_test_long,self.predict_Y_given_X(X_test)))                
      
        for est_y in self.estY:
            if isinstance(est_y,GBM_KClass_MoE):
                est_y.cache = None
                

        if isinstance(self.estZ,GBM_KClass_MoE):
            self.estZ.cache = None
                    
    def predict_Y_given_X(self,X):
        # return P(Y|X) of shape (N,K1)
        P_Z_given_X = self.infer_Z_given_X(X)
        P_Y_given_XZ = self.infer_Y_given_XZ(X)
        return np.einsum('nq,npq->np',P_Z_given_X,P_Y_given_XZ)
    
    def predict_Y_given_X_FixedY(self,X,P_Y_given_XZ):
        # return P(Y|X) of shape (N,K1)
        P_Z_given_X = self.infer_Z_given_X(X)
        return np.einsum('nq,npq->np',P_Z_given_X,P_Y_given_XZ)
    
     
class GBM_KClass_MoE():
    # this class is specifically designed to work with MoE

    def __init__(self,BaseEst,BasePara,baseline,d,learnR,M_est,subFold,subR=0.1):
        self.BaseEst=BaseEst
        self.estimator_=[]
        self.BasePara=BasePara
        self.cache = None # faster training, keep pre-softmax info
        self.baseline = baseline
        self.subFeature = np.random.rand(d) > subR
        self.learnR = learnR
        self.M_est=M_est
        self.subFold=subFold
        
    def update(self,X,grad,index_,non_index):
        # add one base learner
        self.estimator_.append(self.BaseEst(**self.BasePara).fit(X[index_][:,self.subFeature],grad*self.learnR))
        self.cache[non_index] += self.estimator_[-1].predict(X[non_index][:,self.subFeature])
        
        
    def fit(self,X,y,restart=True,M_add=None):

        N = len(y)
        K = len(np.unique(y))
        self.K=K
        Y = y2long(y,K)
        kf = KFold(N, n_folds=self.subFold)
        
        y_raw = Y + self.baseline
        yp = stable_softmax(y_raw)
        if M_add==None:
            M_add=self.M_est
            
        if restart==True:
            self.estimator_=[]
        best_score = []
        scoreFun = lambda y,yhat: np.mean(y*np.log(yhat)) 

                                
        for m in range(M_add):
            index=np.random.permutation(N) # shuffle index for subsampling
            X,Y,yp,y_raw,y = X[index],Y[index],yp[index],y_raw[index],y[index]

            #for test,train in kf:
            for train,test in kf:    
                
                self.estimator_.append(self.BaseEst(**self.BasePara).\
                fit(X[test],(Y[test,:]-yp[test,:])*self.learnR))
                y_raw[train]+=self.estimator_[-1].predict(X[train])
                yp[train]=stable_softmax(y_raw[train])
                best_score.append(scoreFun(Y,yp))
        
        self.M_est=len(self.estimator_)
        plt.plot(best_score)
        return self

        
    def predict_raw(self,X):

        yhat=self.baseline
        for est_ in self.estimator_:
            yhat = yhat + est_.predict(X[:,self.subFeature])
        return yhat       
        
    def predict_proba(self,X,sub_index=None):
        if self.cache is None or sub_index is None:
            return stable_softmax(self.predict_raw(X))
        else:
            return stable_softmax(self.cache[sub_index])
        
    def plot(self,X,y):
        N = len(y)

        accr=np.zeros(self.M_est)
        y_raw=np.copy(self.baseline)
            
        for m in range(self.M_est):
            y_raw= y_raw + self.estimator_[m].predict(X)
            accr[m]=1.0*np.sum(y==np.argmax(y_raw,1))/N
        plt.plot(accr)
        
class LR_KClass_MoE():
    # this class is specifically designed to work with MoE

    def __init__(self,k,d,learnR,subR=0.1):
        self.subFeature = np.random.rand(d) > subR
        self.d = self.subFeature.sum()
        self.k = k
        self.learnR = learnR
        self.beta = np.random.randn(self.d,k) / self.d
        
    def update(self,X,grad):
        n = X.shape[0]
        self.beta += np.dot(X.T[self.subFeature],grad*self.learnR)/n
    
    def fit(self,X,y,iterTimes,batchSize=100):
        n = X.shape[0]        
        batchN = int(n/batchSize)
        Y = y2long(y,self.k)
        for i in range(iterTimes):
            index = np.random.permutation(n)
            for j in range(batchN):
                sub_index = index[j*batchSize:(j+1)*batchSize]
                self.update(X[sub_index], Y[sub_index]-self.predict_proba(X[sub_index]))
        return self
        
    def predict_raw(self,X):
        return np.dot(X[:,self.subFeature], self.beta)
        
    def predict_proba(self,X):
        return stable_softmax(self.predict_raw(X))
