import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from numpy.random import default_rng

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
        return np.exp(np.mean(np.log(individual_predictions),axis=0))

def load_training_data(
    data_directory,
    n_bbh = 20000,
    n_bns = 20000,
    n_nsbh = 20000,
    n_hopeless = 60000,
    n_certain = 20000,
    rng_key = 111):

    generator = default_rng(rng_key)

    # Read injections
    train_data = pd.read_hdf('{0}/bbh_training_data.hdf'.format(data_directory)).sample(n_bbh,random_state=generator)
    val_data = pd.read_hdf('{0}/bbh_validation_data.hdf'.format(data_directory)).sample(int(n_bbh/4),random_state=generator)
    bns_train_data = pd.read_hdf('{0}/bns_training_data.hdf'.format(data_directory)).sample(n_bns,random_state=generator)
    bns_val_data = pd.read_hdf('{0}/bns_validation_data.hdf'.format(data_directory)).sample(int(n_bns/4),random_state=generator)
    nsbh_train_data = pd.read_hdf('{0}/nsbh_training_data.hdf'.format(data_directory)).sample(n_nsbh,random_state=generator)
    nsbh_val_data = pd.read_hdf('{0}/nsbh_validation_data.hdf'.format(data_directory)).sample(int(n_nsbh/4),random_state=generator)

    # Assign class identifiers
    train_data['class'] = 0
    val_data['class'] = 0
    bns_train_data['class'] = 1
    bns_val_data['class'] = 1
    nsbh_train_data['class'] = 2
    nsbh_val_data['class'] = 2

    train_data = pd.concat([train_data,bns_train_data,nsbh_train_data])
    val_data = pd.concat([val_data,bns_val_data,nsbh_val_data])

    # Read and split hopeless injections
    if n_hopeless>0:

        official_hopeless_data = pd.read_hdf('{0}/rpo3-without-hopeless-cut-formatted.hdf'.format(data_directory)).sample(n_hopeless,random_state=generator)
        official_hopeless_data,val_official_hopeless_data = train_test_split(official_hopeless_data,train_size=0.8,random_state=generator.integers(0,high=1024))

        official_hopeless_data['class'] = 3
        val_official_hopeless_data['class'] = 3

        train_data = pd.concat([train_data,official_hopeless_data])
        val_data = pd.concat([val_data,val_official_hopeless_data])

    # Read and split certain injections
    if n_certain>0:

        official_certain_data = pd.read_hdf('{0}/rpo3-certain-formatted.hdf'.format(data_directory)).sample(n_certain,random_state=generator)
        official_certain_data['detected'] = 1
        official_certain_data.loc[official_certain_data['obs_snr']<=8,'detected'] = 0

        # Remove odd events
        to_clip = (official_certain_data.m1_detector<10.*official_certain_data.luminosity_distance**1.2)
        official_certain_data.loc[to_clip,'detected'] = 0

        # Split
        official_certain_data,val_official_certain_data = train_test_split(official_certain_data,train_size=0.8,random_state=generator.integers(0,high=1024))

        official_certain_data['class'] = 4
        val_official_certain_data['class'] = 4

        train_data = pd.concat([train_data,official_certain_data])
        val_data = pd.concat([val_data,val_official_certain_data])

    # Append training and validation sets together and shuffle
    train_data = shuffle(train_data,random_state=generator.integers(0,high=1024))
    val_data = shuffle(val_data,random_state=generator.integers(0,high=1024))

    return train_data,val_data

if __name__=="__main__":
    
    load_training_data(rng_key=12)

