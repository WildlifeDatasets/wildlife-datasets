from .datasets import DatasetFactory
from .datasets import AAUZebraFish
from .datasets import AerialCattle2017
from .datasets import ATRW
from .datasets import BelugaID
from .datasets import BirdIndividualID
from .datasets import BirdIndividualIDSegmented
from .datasets import CTai
from .datasets import CZoo
from .datasets import Cows2021
from .datasets import Drosophila
from .datasets import FriesianCattle2015
from .datasets import FriesianCattle2017
from .datasets import GiraffeZebraID
from .datasets import Giraffes
from .datasets import HappyWhale
from .datasets import HumpbackWhaleID
from .datasets import HyenaID2022
from .datasets import IPanda50
from .datasets import LeopardID2022
from .datasets import LionData
from .datasets import MacaqueFaces
from .datasets import NDD20
from .datasets import NOAARightWhale
from .datasets import NyalaData
from .datasets import OpenCows2020
from .datasets import SealID
from .datasets import SealIDSegmented
from .datasets import SeaTurtleID
from .datasets import SeaTurtleIDHeads
from .datasets import SMALST
from .datasets import StripeSpotter
from .datasets import WhaleSharkID
from .datasets import WNIGiraffes
from .datasets import ZindiTurtleRecall
from .metadata import Metadata

names_all = [
    AAUZebraFish,
    AerialCattle2017,
    ATRW,
    BelugaID,
    BirdIndividualID,
    BirdIndividualIDSegmented,
    CTai,
    CZoo,
    Cows2021,
    Drosophila,
    FriesianCattle2015,
    FriesianCattle2017,
    GiraffeZebraID,
    Giraffes,
    HappyWhale,
    HumpbackWhaleID,
    HyenaID2022,
    IPanda50,
    LeopardID2022,
    LionData,
    MacaqueFaces,
    NDD20,
    NOAARightWhale,
    NyalaData,
    OpenCows2020,
    SealID,
    SealIDSegmented,
    SeaTurtleID,
    SeaTurtleIDHeads,
    SMALST,
    StripeSpotter,
    WhaleSharkID,
    WNIGiraffes,
    ZindiTurtleRecall,
]

names_wild = [
    BelugaID,
    GiraffeZebraID,
    HappyWhale,
    HumpbackWhaleID,
    HyenaID2022,
    LeopardID2022,
    NDD20,
    NOAARightWhale,
    NyalaData,
    SealID,
    SeaTurtleID,
    StripeSpotter,
    WhaleSharkID,
    WNIGiraffes,
]

names_small = [
    AerialCattle2017,
    BelugaID,
    CTai,
    CZoo,
    FriesianCattle2015,
    FriesianCattle2017,
    IPanda50,
    MacaqueFaces,
    NyalaData,
    SeaTurtleIDHeads,
    StripeSpotter,
]

names_cows = [
    AerialCattle2017,
    Cows2021,
    FriesianCattle2015,
    FriesianCattle2017,
    OpenCows2020,
]

names_giraffes = [
    GiraffeZebraID,
    Giraffes,
    SMALST,
    StripeSpotter,
    WNIGiraffes,
]

names_primates = [
    CTai,
    CZoo,
    MacaqueFaces,
]

names_turtles = [
    SeaTurtleIDHeads,
    ZindiTurtleRecall,
]

names_whales = [
    BelugaID,
    HappyWhale,
    HumpbackWhaleID,
    NDD20,
    NOAARightWhale,
    WhaleSharkID,
]
