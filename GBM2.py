
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.cross_validation import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier,ExtraTreeClassifier,ExtraTreeRegressor
import pandas as pd

# from sklearn.ensemble import GradientBoostingClassifier

#class adaBoost_KClass(BaseEstimator, ClassifierMixin):
#    
#    def __init__(self,BaseEst,M_est,learnRate,BasePara,subFold):
#        self.BaseEst=BaseEst
#        self.M_est=M_est
#        self.learnRate=learnRate
#        self.estimator_=[]
#        self.BasePara=BasePara
#        self.subFold=subFold
#        
#    def fit(self,X,y,restart=True,M_add=None,weight=None):
#        # y is a vector of size N
#        # BaseEstimator has to support predict_proba and fit with weight
#        N = len(y)
#        K = len(np.unique(y))
#        self.K=K
#        Y = np.ones((N,K))/(1-K)
#        Y[np.arange(N),y]=1
#        kf = KFold(self.subFold).split(range(N))
#        const=(1-K)/K*self.learnRate
#        
#        if weight==None:
#            w = np.ones(N)/N
#        else:
#            w = weight/np.sum(weight)
#            
#        if M_add==None:
#            M_add=self.M_est
#            
#        if restart==True:
#            self.estimator_=[]
#        else:
#            for m in range(self.M_est):
#                w*=np.exp(const*np.sum(Y*self.estimator_[m].predict_proba(X),1))
#                
#        for m in range(M_add):
#
#            index=np.random.permutation(N) # shuffle index for subsampling
#            X,y,w,Y = X[index],y[index],w[index],Y[index]
#            for train,_ in kf:
#                self.estimator_.append(self.BaseEst(**self.BasePara).\
#                fit(X[train],y[train],sample_weight=w[train]))
#                w*=np.exp(const*np.sum(Y*self.estimator_[-1].predict_proba(X),1))
#                #w=w/np.sum(w)
#        
#        self.M_est=len(self.estimator_)
#        return self
#        
#    def predict(self,X):
#        # predict needs to be a method of base learner
#        yhat=np.zeros((X.shape[0],self.K))
#        for m in range(self.M_est):
#            yhat+=self.estimator_[m].predict_proba(X)
#            #yhat+=self.estimator_[m].predict_log_proba(X)
#        return np.argmax(yhat,1)
#        
#    def predict_proba(self,X):
#        # exp loss is not optimized to calculate prob
#        # predict_proba needs to be a method of base learner
#        yhat=np.zeros((X.shape[0],self.K))
#        for m in range(self.M_est):
#            yhat+=self.estimator_[m].predict_proba(X)
#        temp=np.exp(yhat*self.learnRate)
#        return temp/np.sum(temp,1,keepdims=True)
#        
#    def plot(self,X,y,weight=None):
#        N = len(y)
#        K = self.K
#        accr=np.zeros(self.M_est)
#        yhat=np.zeros((N,K))
#        for m in range(self.M_est):
#            yhat+=self.estimator_[m].predict(X)
#            if weight == None:
#                accr[m]=1.0*np.sum(y==np.argmax(yhat,1))/N
#            else:
#                accr[m]=1.0*np.sum(weight*(y==np.argmax(yhat,1)))
#        plt.plot(accr)
        
        
        


class GBM_KClass(BaseEstimator, ClassifierMixin):
    
    def __init__(self,BaseEst,M_est,learnRate,BasePara,subFold):
        self.BaseEst=BaseEst
        self.M_est=M_est
        self.learnRate=learnRate
        self.estimator_=[]
        self.BasePara=BasePara
        self.subFold=subFold
        # self.subClass=subClass
        
    def fit(self,X,y,weight=None,restart=True,M_add=None,baseline=None):
        # fit one tree with multi-output instead of K trees per step
        # y needs to be a vector of size N and the value of y needs to range
        # from 0 to K-1. The order to yp is the same of value of y, i.e. yp[:,0]
        # is the prob of class 0 and so on
        # y_raw is the baseline model and needs to pass in when refit/predict
        # warmstart is used by passing in previous returned y_raw in baseline arg.
        N = len(y)
        K = len(np.unique(y))
        self.K=K
        Y = np.zeros((N,K))
        Y[np.arange(N),y]=1
        kf = KFold(N, n_folds=self.subFold)
        
        
        if baseline==None:
            p_ = Y.sum(0)/N
            self.baseline = np.log(p_[:K-1]/p_[K-1])
            y_raw=np.zeros((N,K))
            y_raw[:,:K-1] = self.baseline
        else:
            y_raw=np.copy(baseline)
            
        if M_add==None:
            M_add=self.M_est
            
        if restart==True:
            self.estimator_=[]

                     
        temp=np.exp(y_raw)
        yp=temp/np.sum(temp,1,keepdims=True)
        best_score = []
        
        if weight==None:
            # without weight
            for m in range(M_add):
                index=np.random.permutation(N) # shuffle index for subsampling
                X,Y,yp,y_raw,y = X[index],Y[index],yp[index],y_raw[index],y[index]

                #for test,train in kf:
                for train,test in kf:    
                    
                    # subClass
                    # target=Y[test,:]-yp[test,:]
                    # target[np.random.rand(*target.shape)<self.subClass]=0
                    self.estimator_.append(self.BaseEst(**self.BasePara).\
                    fit(X[test],Y[test,:]-yp[test,:]))
                    y_raw[train]+=self.learnRate*self.estimator_[-1].predict(X[train])
                    temp=np.exp(y_raw[train])
                    yp[train]=temp/np.sum(temp,1,keepdims=True)
                    best_score.append(log_loss(y,yp))

        else:
            # with weight
            weight=weight.reshape((N,1))
            for m in range(M_add):
                #pdb.set_trace()
                index=np.random.permutation(N) # shuffle index for subsampling
                X,Y,yp,y_raw,weight,y = X[index],Y[index],yp[index],y_raw[index],weight[index],y[index]
                for train,test in kf:
                    self.estimator_.append(self.BaseEst(**self.BasePara).\
                    fit(X[test,:],(Y[test,:]-yp[test,:])*weight[test]))
                    y_raw[train]+=self.learnRate*self.estimator_[-1].predict(X[train])
                    temp=np.exp(y_raw[train])
                    yp[train]=temp/np.sum(temp,1,keepdims=True)
                    best_score.append(log_loss(y,yp,sample_weight=weight.flatten()))
        
        self.M_est=len(self.estimator_)
        plt.plot(best_score)
        return np.min(best_score), y_raw
        
        
    def predict_raw(self,X,baseline=None):
        if baseline==None:
            yhat=np.zeros((X.shape[0],self.K))
            yhat[:,:self.K-1] = self.baseline
        else:
            yhat=np.copy(baseline)
        for m in range(self.M_est):
            yhat+=self.estimator_[m].predict(X)
        return yhat       
        
    def predict(self,X,baseline=None):
        return np.argmax(self.predict_raw(X,baseline=baseline),1)


    def predict_proba(self,X,baseline=None):
        P=np.exp(self.learnRate*self.predict_raw(X,baseline=baseline))
        return P/np.sum(P,1,keepdims=True)
    
        
    def score(self,X,y,baseline=None,score_fun=accuracy_score,IsProb=False):
        if IsProb:
            return score_fun(y,self.predict_proba(X,baseline=baseline))
        else:
            return score_fun(y,self.predict(X,baseline=baseline))
        
    def plot(self,X,y,weight=None,baseline=None):
        N = len(y)
        K = self.K
        accr=np.zeros(self.M_est)
        if baseline==None:
            y_raw=np.zeros((N,K))
            y_raw[:,:self.K-1] = self.baseline
        else:
            y_raw=np.copy(baseline)
        
        if weight!=None:
            weight=weight/np.sum(weight)
            
        for m in range(self.M_est):
            y_raw+=self.estimator_[m].predict(X)
            if weight == None:
                accr[m]=1.0*np.sum(y==np.argmax(y_raw,1))/N
            else:
                accr[m]=1.0*np.sum(weight*(y==np.argmax(y_raw,1)))
        plt.plot(accr)
        
    def plot_MLE(self,X,y,weight=None,baseline=None):
        N = len(y)
        K = self.K
        accr=np.zeros(self.M_est)
        Y = np.zeros((N,K))
        Y[np.arange(N),y]=1
        
        if baseline==None:
            y_raw=np.zeros((N,K))
            y_raw[:,:self.K-1] = self.baseline
        else:
            y_raw=np.copy(baseline)
            
        if weight!=None:
            weight=weight/np.sum(weight)
        for m in range(self.M_est):
            y_raw+=self.learnRate*self.estimator_[m].predict(X)
            temp=np.exp(y_raw)
            yp=temp/np.sum(temp,1,keepdims=True)
            if weight == None:
                accr[m]=np.mean(np.log(np.sum(Y*yp,1)))
            else:
                accr[m]=np.sum(np.log(np.sum(Y*yp,1))*weight)
        plt.plot(accr)

        
        
def GBM_RandomSearch(X,y,Ntry,M,FixPara,RandomPara,weight):
    # FixPara is a dict of fixed para
    # RandomPara is a list of tuple para for randomFun
    
    result = np.zeros((Ntry,4))
    
    for i in range(Ntry):
        RanPara={}
        result[i,0] = np.random.randint(*RandomPara[0])
        RanPara['subFold'] = result[i,0]
        
        BasePara={}
        result[i,1] = np.random.randint(*RandomPara[1])
        BasePara['max_depth'] = result[i,1]
        result[i,2] = np.random.uniform(*RandomPara[2])
        BasePara['max_features'] = result[i,2]        
        RanPara['BasePara']=BasePara
        
    
        
        RanPara.update(FixPara)
        RanPara['M_est'] = int(M/RanPara['subFold'])
        #pdb.set_trace()
        model = GBM_KClass(**RanPara)
        result[i,3] = model.fit(X,y,weight)

        
    return pd.DataFrame(result,columns=['subFold','max_depth','max_features','acc'])
            
    
    
    

    
#model1=GBM_KClass(ExtraTreeRegressor,100,0.005,{'max_depth':16,'splitter':'random','max_features':0.9},2)
#model1.fit(X_train,y_train)

paras1 = {'BaseEst':ExtraTreeRegressor,'M_est':100, 'learnRate':0.0001}
paras2 = {'BasePara':{'max_depth':16,'splitter':'random','max_features':0.9},'subFold':2}



def LossMask(Y,P,mask):
    # Y should have shape (n,N), P (N,d) and mask (n,N)
    return (np.dot(Y*mask,np.log(P)) + np.dot((1-Y)*mask,np.log(1-P)))/np.sum(mask,1,keepdims=True)

def dataGenMaskY(n,N,p1,p2,P):
    # P should be of shape (N,d) and serves as hyper-para
    # p1 is the prob for Y, p2 for mask
    while True:
        Y = np.random.rand(n,N)>p1
        mask = np.random.rand(n,N)>p2
        C = LossMask(Y,P,mask)
        ret = np.zeros((n,N,2),dtype=np.int32) # 3 classes, K-1 =2
        ret[:,:,0] = 1-Y
        ret[:,:,1] = Y
        ret = ret * mask[:,:,np.newaxis]
        yield (C,np.reshape(ret,(n,N*2)))

def sigmoid(x):
    return 1/(1+np.exp(-x))


    
class GBM_KD(BaseEstimator, ClassifierMixin):
    # predict D output, each having K classes. So the total number of output
    # will be K*D. training data is provided in the form of a generator.
    
    def __init__(self,BaseEst,M_est,learnRate,BasePara,K,D,baseline):
        self.BaseEst=BaseEst
        self.M_est=M_est
        self.learnRate=learnRate
        self.estimator_=[]
        self.BasePara=BasePara
        self.K = K
        self.D = D
        self.baseline = baseline # should be of the shape (K*D,)
    
    @staticmethod    
    def logLoss(y,yp):
        return np.mean(y*np.log(yp))
    
    def fit(self,dataGen,restart=True,M_add=None):
          
        if M_add==None:
            M_add=self.M_est
            
        if restart==True:
            self.estimator_=[]

        for m in range(M_add):
            X,Y = dataGen.next()
            yp = self.predict_proba(X)
            print "iteration:{}, logLoss:{}".format(m,self.logLoss(Y,yp)) # partial loss. Does not have class mask.
            self.estimator_.append(self.BaseEst(**self.BasePara).fit(X,Y-yp))
            yp = self.predict_proba(X)
            print "iteration:{}, logLoss:{}".format(m,self.logLoss(Y,yp))
            
        self.M_est=len(self.estimator_)
        return self
        
        
    def predict_raw(self,X):
        yhat=np.copy(self.baseline)
        for m in self.estimator_:
            yhat=yhat+m.predict(X)
        return yhat       
        
    def predict_class(self,X):
        return self.predict_raw(X)>0

    def predict_proba(self,X):
        temp = np.exp(np.reshape(self.learnRate*self.predict_raw(X),(-1,self.D,self.K-1)))
        return np.reshape(temp/(np.sum(temp,2,keepdims=True)+1),(-1,self.D*(self.K-1)))
         
    def plot_MLE(self,X,y):
        accr=np.zeros(self.M_est)        
        y_raw=np.copy(self.baseline)
            
        for m in range(self.M_est):
            y_raw=y_raw + self.learnRate*self.estimator_[m].predict(X)
            yp=sigmoid(y_raw)
            accr[m]=self.logLoss(y,yp)
        plt.plot(accr)































