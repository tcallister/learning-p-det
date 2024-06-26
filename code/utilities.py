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
    n_bbh_certain = 20000,
    n_bbh_hopeless = 60000,
    n_bns_certain = 20000,
    n_bns_hopeless = 20000,
    n_nsbh_certain = 20000,
    n_nsbh_hopeless = 20000,
    n_combined_hopeless = 20000,
    rng_key = 111):

    generator = default_rng(rng_key)

    ##### TEST
    ##### Sample and reweight
    """
    bbh_train_data_raw = pd.read_hdf('{0}/bbh_training_data.hdf'.format(data_directory))
    zs = bbh_train_data_raw.redshift
    m1_det = bbh_train_data_raw.m1_detector
    m1_src = m1_det/(1.+zs)
    #m1 = bbh_train_data_raw.pri
    ws = (1.+zs)**2/m1_src
    print(np.sum(ws)**2/np.sum(ws**2))
    bbh_train_data = bbh_train_data_raw.sample(n_bbh,weights=ws,random_state=generator)
    """

    # Read injections
    bbh_train_data = pd.read_hdf('{0}/bbh_training_data.hdf'.format(data_directory)).sample(n_bbh,random_state=generator)
    bns_train_data = pd.read_hdf('{0}/bns_training_data.hdf'.format(data_directory)).sample(n_bns,random_state=generator)
    nsbh_train_data = pd.read_hdf('{0}/nsbh_training_data.hdf'.format(data_directory)).sample(n_nsbh,random_state=generator)

    #print(pd.read_hdf('{0}/bbh_training_data.hdf'.format(data_directory)).shape)

    # Read pre-prepared validation sets
    bbh_val_data = pd.read_hdf('{0}/bbh_validation_data.hdf'.format(data_directory)).sample(int(n_bbh/4),random_state=generator)
    bns_val_data = pd.read_hdf('{0}/bns_validation_data.hdf'.format(data_directory)).sample(int(n_bns/4),random_state=generator)
    nsbh_val_data = pd.read_hdf('{0}/nsbh_validation_data.hdf'.format(data_directory)).sample(int(n_nsbh/4),random_state=generator)

    # Concat together
    train_data = pd.concat([bbh_train_data,bns_train_data,nsbh_train_data])
    val_data = pd.concat([bbh_val_data,bns_val_data,nsbh_val_data])

    # Read and split hopeless injections
    # Hopeless BBH
    if n_bbh_hopeless>0:
        bbh_hopeless_data = pd.read_hdf('{0}/rpo3-bbh-hopeless-formatted.hdf'.format(data_directory)).sample(n_bbh_hopeless,random_state=generator)
        bbh_hopeless_data,val_bbh_hopeless_data = train_test_split(bbh_hopeless_data,train_size=0.8,random_state=generator.integers(0,high=1024))
        train_data = pd.concat([train_data,bbh_hopeless_data])
        val_data = pd.concat([val_data,val_bbh_hopeless_data])

    #print(pd.read_hdf('{0}/rpo3-bbh-hopeless-formatted.hdf'.format(data_directory)).shape)

    # Hopeless BNS
    if n_bns_hopeless>0:
        bns_hopeless_data = pd.read_hdf('{0}/rpo3-bns-hopeless-formatted.hdf'.format(data_directory)).sample(n_bns_hopeless,random_state=generator)
        bns_hopeless_data,val_bns_hopeless_data = train_test_split(bns_hopeless_data,train_size=0.8,random_state=generator.integers(0,high=1024))
        train_data = pd.concat([train_data,bns_hopeless_data])
        val_data = pd.concat([val_data,val_bns_hopeless_data])

    # Hopeless NSBH
    if n_nsbh_hopeless>0:
        nsbh_hopeless_data = pd.read_hdf('{0}/rpo3-nsbh-hopeless-formatted.hdf'.format(data_directory)).sample(n_nsbh_hopeless,random_state=generator)
        nsbh_hopeless_data,val_nsbh_hopeless_data = train_test_split(nsbh_hopeless_data,train_size=0.8,random_state=generator.integers(0,high=1024))
        train_data = pd.concat([train_data,nsbh_hopeless_data])
        val_data = pd.concat([val_data,val_nsbh_hopeless_data])

    # Hopeless auxiliary
    if n_combined_hopeless>0:
        combined_hopeless_data = pd.read_hdf('{0}/rpo3-combined-hopeless-formatted.hdf'.format(data_directory)).sample(n_combined_hopeless,random_state=generator)
        combined_hopeless_data,val_combined_hopeless_data = train_test_split(combined_hopeless_data,train_size=0.8,random_state=generator.integers(0,high=1024))
        train_data = pd.concat([train_data,combined_hopeless_data])
        val_data = pd.concat([val_data,val_combined_hopeless_data])

    print(pd.read_hdf('{0}/rpo3-combined-hopeless-formatted.hdf'.format(data_directory)).shape)

    # Read and split certain injections
    # Certain BBH
    if n_bbh_certain>0:

        bbh_certain_data = pd.read_hdf('{0}/rpo3-bbh-certain-formatted.hdf'.format(data_directory)).sample(n_bbh_certain,random_state=generator)

        # Remove odd events
        to_clip = (bbh_certain_data.chirp_mass_detector<10.*bbh_certain_data.luminosity_distance**1.2)
        bbh_certain_data = bbh_certain_data.drop(bbh_certain_data.loc[to_clip,:].index)

        # Split and add to training data
        bbh_certain_data,val_bbh_certain_data = train_test_split(bbh_certain_data,train_size=0.8,random_state=generator.integers(0,high=1024))
        train_data = pd.concat([train_data,bbh_certain_data])
        val_data = pd.concat([val_data,val_bbh_certain_data])

    print(pd.read_hdf('{0}/rpo3-bbh-certain-formatted.hdf'.format(data_directory)).shape)

    # Certain BNS
    if n_bns_certain>0:

        bns_certain_data = pd.read_hdf('{0}/rpo3-bns-certain-formatted.hdf'.format(data_directory)).sample(n_bns_certain,random_state=generator)

        # Split and add to training data
        bns_certain_data,val_bns_certain_data = train_test_split(bns_certain_data,train_size=0.8,random_state=generator.integers(0,high=1024))
        train_data = pd.concat([train_data,bns_certain_data])
        val_data = pd.concat([val_data,val_bns_certain_data])

    # Certain NSBH
    if n_nsbh_certain>0:

        nsbh_certain_data = pd.read_hdf('{0}/rpo3-nsbh-certain-formatted.hdf'.format(data_directory)).sample(n_nsbh_certain,random_state=generator)

        # Remove odd events
        to_clip = (nsbh_certain_data.chirp_mass_detector<(nsbh_certain_data.luminosity_distance/0.1)**1.4)
        nsbh_certain_data = nsbh_certain_data.drop(nsbh_certain_data.loc[to_clip,:].index)

        # Split and add to training data
        nsbh_certain_data,val_nsbh_certain_data = train_test_split(nsbh_certain_data,train_size=0.8,random_state=generator.integers(0,high=1024))
        train_data = pd.concat([train_data,nsbh_certain_data])
        val_data = pd.concat([val_data,val_nsbh_certain_data])

    # Shuffle
    train_data = shuffle(train_data,random_state=generator.integers(0,high=1024))
    val_data = shuffle(val_data,random_state=generator.integers(0,high=1024))

    return train_data,val_data

if __name__=="__main__":
    
    load_training_data(data_directory='/project/kicp/tcallister/learning-p-det-data/input_data/',rng_key=12)

