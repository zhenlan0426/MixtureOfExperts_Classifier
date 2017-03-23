import numpy as np

def stable_softmax(X):
    temp = np.exp(X + np.mean(X,1,keepdims=True))
    return temp/np.sum(temp,1,keepdims=True)

def y2long(Y,k):
    n = Y.shape[0]
    Y_long = np.zeros((n,k))
    Y_long[range(n),Y] = 1
    return Y_long

class MixtureOfExperts_Classifier():
    # implement MLAPP,11.2.4 Mixtures of experts
    
    def __init__(self,K1,K2,d):
        self.K1 = K1 # for Y
        self.K2 = K2 # for Z, hyper-parameter
        self.d = d
        self.beta = np.random.randn(d,K1,K2)/d 
        self.theta = np.random.randn(d,K2)/d
    
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
        # do one gradient decent step
        
        # update theta
        n = X.shape[0]
        Z_givenXY, Z_givenX, Y_givenXZ,Like_Y_givenXZ = self.infer_Z_given_XY(X,Y,returnAll=True)
        self.theta += learnR * np.dot(X.T,Z_givenXY-Z_givenX)/n 
        
        # update beta
        Y_long = y2long(Y,self.K1)
        self.beta += learnR * np.einsum('nd,npq->dpq',X,np.einsum('np,nq->npq',Y_long,Like_Y_givenXZ)-Y_givenXZ)/n
        
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
n = 10000
d = 10
k1 = 5
k2 = 3

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
'''










