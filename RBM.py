import pickle
import sklearn
from scipy.special import expit
import numpy as np

def get_RBM(frame):

    # load the best basic RBM model
    rbm = pickle.load(open('RBM_model0.sav', 'rb'))

    # resize for calculations
    frame2 = np.reshape(frame, 4096)

    # getting RBMs internal values
    hidden_biases = rbm.intercept_hidden_
    visible_biases = rbm.intercept_visible_
    weights = rbm.components_

    # forward pass
    PHiV0 = expit(weights.dot(frame2) + hidden_biases)

    # backward pass
    PH0Vi = expit(weights.T.dot(PHiV0) + visible_biases)

    # resize to original dimensions for display
    PH0Vi = np.reshape(PH0Vi, (64, 64))

    return PH0Vi

def get_DRBM(frame):

    # resize for calculations
    frame2 = np.reshape(frame, 4096)

    # load the top layer and internal values
    rbm = pickle.load(open('DRBM_model0_1.sav', 'rb'))
    hidden_biases = rbm.intercept_hidden_
    visible_biases = rbm.intercept_visible_
    weights = rbm.components_

    # load the bottom layer and internal values
    rbm2 = pickle.load(open('DRBM_model0_2.sav', 'rb'))
    hidden_biases2 = rbm2.intercept_hidden_
    visible_biases2 = rbm2.intercept_visible_
    weights2 = rbm2.components_

    # forward pass - top layer
    PHiV0 = expit(weights.dot(frame2) + hidden_biases)

    # forward pass - bottom layer
    inter_in = expit(weights2.dot(PHiV0) + hidden_biases2)

    # backward pass - bottom layer
    inter_out = expit(weights2.T.dot(inter_in) + visible_biases2)

    # backward pass - top layer
    PH0Vi = expit(weights.T.dot(inter_out) + visible_biases)

    # resize to original dimensions for display
    PH0Vi = np.reshape(PH0Vi, (64, 64))

    return PH0Vi

def get_stacked(frame):

    # resize for calculations
    frame2 = np.reshape(frame, 4096)

    # top layer
    rbm = pickle.load(open('DRBM_model2_1.sav', 'rb'))
    hidden_biases = rbm.intercept_hidden_
    visible_biases = rbm.intercept_visible_
    weights = rbm.components_

    # middle layer
    rbm2 = pickle.load(open('DRBM_model2_2.sav', 'rb'))
    hidden_biases2 = rbm2.intercept_hidden_
    visible_biases2 = rbm2.intercept_visible_
    weights2 = rbm2.components_

    # bottom layer
    rbm3 = pickle.load(open('DRBM_model2_3.sav', 'rb'))
    hidden_biases3 = rbm3.intercept_hidden_
    visible_biases3 = rbm3.intercept_visible_
    weights3 = rbm3.components_

    # forward pass 1
    PHiV0 = expit(weights.dot(frame2) + hidden_biases)
    PH0Vi = expit(weights.T.dot(PHiV0) + visible_biases)

    # forward pass 2
    PHiV02 = expit(weights2.dot(PH0Vi) + hidden_biases2)
    PH0Vi2 = expit(weights2.T.dot(PHiV02) + visible_biases2)
    
    # forward pass 3
    PHiV03 = expit(weights3.dot(PH0Vi2) + hidden_biases3)
    PH0Vi3 = expit(weights3.T.dot(PHiV03) + visible_biases3)

    # resize to original dimensions for display
    PH0Vi3 = np.reshape(PH0Vi3, (64, 64))

    return PH0Vi3