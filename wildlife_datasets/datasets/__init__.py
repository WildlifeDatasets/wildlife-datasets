from .datasets import DatasetFactory
from .aau_zebrafish import AAUZebraFish
from .aerial_cattle import AerialCattle2017
from .atrw import ATRW
from .amvrakikos_turtles import AmvrakikosTurtles
from .beluga_id import BelugaID, BelugaIDv2
from .bird_individual_id import BirdIndividualID, BirdIndividualIDSegmented
from .cat_individual_images import CatIndividualImages
from .ctai import CTai
from .czoo import CZoo
from .cow_dataset import CowDataset
from .cows import Cows2021, Cows2021v2
from .dog_face_net import DogFaceNet
from .drosophila import Drosophila
from .elpephants import ELPephants
from .friesian_cattle import FriesianCattle2015, FriesianCattle2015v2, FriesianCattle2017
from .giraffe_zebra_id import GiraffeZebraID
from .giraffes import Giraffes
from .green_sea_turtles import GreenSeaTurtles
from .happy_whale import HappyWhale
from .humpback_whale_id import HumpbackWhaleID
from .hyena_id import HyenaID2022
from .ipanda import IPanda50
from .leopard_id import LeopardID2022
from .lion_data import LionData
from .macaque_faces import MacaqueFaces
from .mpdd import MPDD
from .ndd import NDD20, NDD20v2
from .noaa_right_whale import NOAARightWhale
from .nyala_data import NyalaData
from .open_cows import OpenCows2020
from .polar_bear_vid_id import PolarBearVidID
from .reunion_turtles import ReunionTurtles
from .seal_id import SealID, SealIDSegmented
from .sea_star_reid import SeaStarReID2023
from .sea_turtle_id import SeaTurtleID2022, SeaTurtleIDHeads
from .smalst import SMALST
from .stripe_spotter import StripeSpotter
from .wildlife_reid_10k import WildlifeReID10k
from .whaleshark_id import WhaleSharkID
from .zakynthos_turtles import ZakynthosTurtles
from .zindi_turtle_recall import ZindiTurtleRecall
from .summary import Summary
from .utils import get_image, load_image

names_all = [
    AAUZebraFish,
    AerialCattle2017,
    AmvrakikosTurtles,
    ATRW,
    BelugaIDv2,
    BirdIndividualID, BirdIndividualIDSegmented,
    CatIndividualImages,
    CTai,
    CZoo,
    CowDataset,
    Cows2021v2,
    DogFaceNet,
    Drosophila,
    ELPephants,
    FriesianCattle2015v2,
    FriesianCattle2017,
    GiraffeZebraID,
    Giraffes,
    GreenSeaTurtles,
    HappyWhale,
    HumpbackWhaleID,
    HyenaID2022,
    IPanda50,
    LeopardID2022,
    LionData,
    MacaqueFaces,
    MPDD,
    NDD20v2,
    NOAARightWhale,
    NyalaData,
    OpenCows2020,
    PolarBearVidID,
    ReunionTurtles,
    SealID, SealIDSegmented,
    SeaStarReID2023,
    SeaTurtleID2022, SeaTurtleIDHeads,
    SMALST,
    StripeSpotter,
    WhaleSharkID,
    ZakynthosTurtles,
    ZindiTurtleRecall,
]

names_wild = [
    AmvrakikosTurtles,
    BelugaIDv2,
    ELPephants,
    GiraffeZebraID,
    GreenSeaTurtles,
    HappyWhale,
    HumpbackWhaleID,
    HyenaID2022,
    LeopardID2022,
    NDD20v2,
    NOAARightWhale,
    NyalaData,
    ReunionTurtles,
    SealID,
    SeaTurtleID2022,
    StripeSpotter,
    WhaleSharkID,
    ZakynthosTurtles,
]

names_small = [
    AerialCattle2017,
    BelugaIDv2,
    CTai,
    CZoo,
    DogFaceNet,
    ELPephants,
    FriesianCattle2015v2,
    FriesianCattle2017,
    IPanda50,
    GreenSeaTurtles,
    MacaqueFaces,
    MPDD,
    NyalaData,
    PolarBearVidID,
    ReunionTurtles,
    SeaTurtleIDHeads,
    StripeSpotter,
    ZakynthosTurtles,
]

names_cows = [
    AerialCattle2017,
    CowDataset,
    Cows2021v2,
    FriesianCattle2015v2,
    FriesianCattle2017,
    OpenCows2020,
]

names_dogs = [
    DogFaceNet,
    MPDD,
]

names_giraffes = [
    GiraffeZebraID,
    Giraffes,
    SMALST,
    StripeSpotter,
]

names_primates = [
    CTai,
    CZoo,
    MacaqueFaces,
]

names_turtles = [
    AmvrakikosTurtles,
    GreenSeaTurtles,
    ReunionTurtles,
    SeaTurtleIDHeads,
    ZakynthosTurtles,
    ZindiTurtleRecall,
]

names_whales = [
    BelugaIDv2,
    HappyWhale,
    HumpbackWhaleID,
    NDD20v2,
    NOAARightWhale,
    WhaleSharkID,
]
