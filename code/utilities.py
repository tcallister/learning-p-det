import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def generalized_Xp(s1x,s1y,s2x,s2y,q):

    """
    Function to compute generalized effective precessing spin parameter,
    as defined in Gerosa+ 2021 (https://arxiv.org/abs/2011.11948)

    Parameters
    ----------
    s1x : `float` or `np.array`
        x-component of the primary's spin
    s1y : `float` or `np.array`
        y-component of the primary's spin
    s2x : `float` or `np.array`
        x-component of the secondary's spin
    s2y : `float` or `np.array`
        y-component of the secondary's spin
    q : `float` or `np.array`

    Returns
    -------
    Xp : `float` or `np.array`
        Generalized precessing spin parameter
    """
    
    # Get total in-plane component spins
    Xp1 = np.sqrt(s1x**2+s1y**2)
    Xp2 = np.sqrt(s2x**2+s2y**2)

    # Cosine of angle between component spins, after projection on the orbital plane
    Xp1_Xp2_cos_dphi = (s1x*s2x + s1y*s2y)

    # Compute precessing spin parameter
    Xp = np.sqrt(Xp1**2 + ((3.+4.*q)/(4.+3.*q)*q*Xp2)**2 + 2.*q*(3.+4.*q)/(4.+3.*q)*Xp1_Xp2_cos_dphi)

    return Xp

class ANNaverage():

    """
    Class used as a wrapper around a list of individually-trained neural networks.
    Used to average the predictions across these individual networks to increase overall accuracy.
    """
    
    def __init__(self,ann_list):

        # List of `tf.keras.models.Sequential` neural networks
        self.ann_list = ann_list
        
    def predict(self,params,*args,**kwargs):

        """
        Function to compute predictions across the elements of `self.ann_list` and return their average.

        Parameters
        ----------
        params : `list`
            Input parameters at which to predict outputs

        Returns
        -------
        mean_predictions : `list`
            Mean predictions, taken across `self.ann_list`
        """

        # Compute list of predictions from individual networks, passing any additional arguments
        individual_predictions = [ann.predict(params,*args,**kwargs) for ann in self.ann_list]

        # Compute and return mean!
        #return np.exp(np.mean(np.log(individual_predictions),axis=0))
        return np.mean(individual_predictions,axis=0)

def load_training_data(
    data_directory,
    n_bbh = 20000,
    n_bns = 20000,
    n_nsbh = 20000,
    n_hopeless = 60000,
    rng_key = 111):

    # Read injections
    train_data = pd.read_hdf('{0}/bbh_training_data.hdf'.format(data_directory)).sample(n_bbh,random_state=rng_key)
    val_data = pd.read_hdf('{0}/bbh_validation_data.hdf'.format(data_directory)).sample(int(n_bbh/4),random_state=rng_key)
    bns_train_data = pd.read_hdf('{0}/bns_training_data.hdf'.format(data_directory)).sample(n_bns,random_state=rng_key)
    bns_val_data = pd.read_hdf('{0}/bns_validation_data.hdf'.format(data_directory)).sample(int(n_bns/4),random_state=rng_key)
    nsbh_train_data = pd.read_hdf('{0}/nsbh_training_data.hdf'.format(data_directory)).sample(n_nsbh,random_state=rng_key)
    nsbh_val_data = pd.read_hdf('{0}/nsbh_validation_data.hdf'.format(data_directory)).sample(int(n_nsbh/4),random_state=rng_key)

    # Read and split hopeless injections
    official_hopeless_data = pd.read_hdf('{0}/rpo3-without-hopeless-cut-formatted.hdf'.format(data_directory)).sample(n_hopeless,random_state=rng_key)
    official_hopeless_data,val_official_hopeless_data = train_test_split(official_hopeless_data,train_size=0.8,random_state=rng_key)

    # Append training and validation sets together and shuffle
    train_data = shuffle(train_data.append(official_hopeless_data).\
                            append(bns_train_data).\
                            append(nsbh_train_data),\
                            random_state=rng_key
                        )
    val_data = shuffle(val_data.append(val_official_hopeless_data).\
                           append(bns_val_data).\
                           append(nsbh_val_data),
                           random_state=rng_key
                       )

    return train_data,val_data

if __name__=="__main__":
    
    load_training_data(rng_key=12)

