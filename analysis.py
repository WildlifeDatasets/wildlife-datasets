import os
import pickle
from datasets import *

datasets_list = [
    (AAUZebraFishID, {}),
    (AerialCattle2017, {}),
    (ATRW, {}),
    (BelugaID, {}),
    (BirdIndividualID, {'variant': 'source'}),
    (BirdIndividualID, {'variant': 'segmented'}),
    (CTai, {}),
    (CZoo, {}),
    (Cows2021, {}),
    (Drosophila, {}),
    (FriesianCattle2015, {}),
    (FriesianCattle2017, {}),
    (GiraffeZebraID, {}),
    (Giraffes, {}),
    (HappyWhale, {}),
    (HumpbackWhaleID, {}),
    (HyenaID2022, {}),
    (IPanda50, {}),
    (LeopardID2022, {}),
    (LionData, {}),
    (MacaqueFaces, {}),
    (NDD20, {}),
    (NOAARightWhale, {}),
    (NyalaData, {}),
    (OpenCows2020, {}),
    (SealID, {'variant': 'source'}),
    (SealID, {'variant': 'segmented'}),
    (SMALST, {}),
    (StripeSpotter, {}),
    (WhaleSharkID, {}),
    (WNIGiraffes, {}),
    (ZindiTurtleRecall, {})
]

class Method:
    def __init__(self, name, root_dataset, root_dataframe=None, variant=None):
        self.name = name
        self.variant = variant
        self.root_dataset = root_dataset
        self.root_dataframe = root_dataframe
        if variant is None:
            self.save_name = name.__name__
        else:
            self.save_name = name.__name__ + '_' + variant

    def create_dataset(self):
        root = os.path.join(self.root_dataset, self.name.__name__)
        if self.variant is None:
            self.dataset = self.name(root)
        else:
            self.dataset = self.name(root, variant=self.variant)

    def save_dataset(self):
        with open(self.get_file_name(), 'wb') as handle:
            pickle.dump(self.dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_dataset(self):
        with open(self.get_file_name(), 'rb') as handle:
            self.dataset = pickle.load(handle)

    def load_or_create_dataset(self, overwrite=False, save=True):
        file_name = self.get_file_name()
        if overwrite or not os.path.exists(file_name):
            self.create_dataset()
            if save:
                self.save_dataset()
        else:
            self.load_dataset()
        
    def get_file_name(self):
        return os.path.join(self.root_dataframe, self.save_name + '.pkl')

