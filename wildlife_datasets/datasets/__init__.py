from .datasets import DatasetFactory, WildlifeDataset
from .downloads import DownloadHuggingFace, DownloadKaggle, DownloadURL, DownloadINaturalist
from .aau_zebrafish import AAUZebraFish
from .aerial_cattle import AerialCattle2017
from .amvrakikos_turtles import AmvrakikosTurtles
from .animal_clef import AnimalCLEF2025, AnimalCLEF2025_LynxID2025, AnimalCLEF2025_SalamanderID2025, AnimalCLEF2025_SeaTurtleID2022
from .atrw import ATRW
from .beluga_id import BelugaID, BelugaIDv2
from .bird_individual_id import BirdIndividualID, BirdIndividualIDSegmented
from .bristol_gorillas_2020 import BristolGorillas2020
from .cattle_muzzle import CattleMuzzle
from .cat_individual_images import CatIndividualImages
from .cobra_re_identification_youngstock import CoBRAReIdentificationYoungstock
from .ctai import CTai
from .czoo import CZoo
from .chicks4free_id import Chicks4FreeID
from .cow_dataset import CowDataset
from .cows import Cows2021, Cows2021v2
from .czechlynx import CzechLynx
from .dog_face_net import DogFaceNet
from .drosophila import Drosophila
from .elpephants import ELPephants
from .friesian_cattle import FriesianCattle2015, FriesianCattle2015v2, FriesianCattle2017
from .giraffe_zebra_id import GiraffeZebraID
from .giraffes import Giraffes
from .happy_whale import HappyWhale
from .holstein_cattle_recognition import HolsteinCattleRecognition
from .humpback_whale_id import HumpbackWhaleID
from .hyena_id import HyenaID2022
from .ipanda import IPanda50
from .leopard_id import LeopardID2022
from .lion_data import LionData
from .macaque_faces import MacaqueFaces
from .mpdd import MPDD
from .multi_cam_cows import MultiCamCows2024
from .ndd import NDD20, NDD20v2
from .noaa_right_whale import NOAARightWhale
from .nyala_data import NyalaData
from .open_cows import OpenCows2020
from .polar_bear_vid_id import PolarBearVidID
from .prim_face import PrimFace
from .reunion_turtles import ReunionTurtles
from .seal_id import SealID, SealIDSegmented
from .sea_star_reid import SeaStarReID2023
from .sea_turtle_id import SeaTurtleID2022, SeaTurtleIDHeads
from .smalst import SMALST
from .southern_province_turtles import SouthernProvinceTurtles
from .stripe_spotter import StripeSpotter
from .wildlife_reid_10k import WildlifeReID10k
from .wild_raptor_id import WildRaptorID
from .whaleshark_id import WhaleSharkID
from .zakynthos_turtles import ZakynthosTurtles
from .zindi_turtle_recall import ZindiTurtleRecall
from .turtles_of_smsrc import TurtlesOfSMSRC 
from .utils import get_image, load_image

names_all = [
    AAUZebraFish,
    AerialCattle2017,
    AmvrakikosTurtles,
    ATRW,
    BelugaIDv2,
    BirdIndividualID, BirdIndividualIDSegmented,
    BristolGorillas2020,
    CattleMuzzle,
    CatIndividualImages,
    Chicks4FreeID,
    CoBRAReIdentificationYoungstock,
    CowDataset,
    Cows2021v2,
    CzechLynx,
    CTai,
    CZoo,
    DogFaceNet,
    Drosophila,
    ELPephants,
    FriesianCattle2015v2,
    FriesianCattle2017,
    GiraffeZebraID,
    Giraffes,
    HappyWhale,
    HolsteinCattleRecognition,
    HumpbackWhaleID,
    HyenaID2022,
    IPanda50,
    LeopardID2022,
    LionData,
    MacaqueFaces,
    MPDD,
    MultiCamCows2024,
    NDD20v2,
    NOAARightWhale,
    NyalaData,
    OpenCows2020,
    PolarBearVidID,
    PrimFace,
    ReunionTurtles,
    SealID, SealIDSegmented,
    SeaStarReID2023,
    SeaTurtleID2022, SeaTurtleIDHeads,
    SMALST,
    SouthernProvinceTurtles,
    StripeSpotter,
    WildRaptorID,
    WhaleSharkID,
    ZakynthosTurtles,
    ZindiTurtleRecall,
    TurtlesOfSMSRC,
]

names_wild = [
    AmvrakikosTurtles,
    BelugaIDv2,
    CzechLynx,
    ELPephants,
    GiraffeZebraID,
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
    SouthernProvinceTurtles,
    StripeSpotter,
    WildRaptorID,
    WhaleSharkID,
    ZakynthosTurtles,
]

names_small = [
    AerialCattle2017,
    BelugaIDv2,
    CoBRAReIdentificationYoungstock,
    CTai,
    CZoo,
    DogFaceNet,
    ELPephants,
    FriesianCattle2015v2,
    FriesianCattle2017,
    HolsteinCattleRecognition,
    IPanda50,
    MacaqueFaces,
    MPDD,
    NyalaData,
    PolarBearVidID,
    ReunionTurtles,
    SeaTurtleIDHeads,
    SouthernProvinceTurtles,
    StripeSpotter,
    ZakynthosTurtles,
]

names_cows = [
    AerialCattle2017,
    CattleMuzzle,
    CoBRAReIdentificationYoungstock,
    CowDataset,
    Cows2021v2,
    FriesianCattle2015v2,
    FriesianCattle2017,
    HolsteinCattleRecognition,
    MultiCamCows2024,
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
    BristolGorillas2020,
    CTai,
    CZoo,
    MacaqueFaces,
    PrimFace,
]

names_turtles = [
    AmvrakikosTurtles,
    ReunionTurtles,
    SeaTurtleIDHeads,
    SouthernProvinceTurtles,
    ZakynthosTurtles,
    ZindiTurtleRecall,
    TurtlesOfSMSRC,
]

names_whales = [
    BelugaIDv2,
    HappyWhale,
    HumpbackWhaleID,
    NDD20v2,
    NOAARightWhale,
    WhaleSharkID,
]
