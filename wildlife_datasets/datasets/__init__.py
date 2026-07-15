from .aau_zebrafish import AAUZebraFish
from .aerial_cattle import AerialCattle2017
from .amvrakikos_turtles import AmvrakikosTurtles
from .animal_clef import (
    AnimalCLEF2025,
    AnimalCLEF2025_LynxID2025,
    AnimalCLEF2025_SalamanderID2025,
    AnimalCLEF2025_SeaTurtleID2022,
    AnimalCLEF2026,
    AnimalCLEF2026_TexasHornedLizards,
)
from .atrw import ATRW
from .balearic_lizard import BalearicLizard, BalearicLizardSegmented
from .beluga_id import BelugaID, BelugaIDv2
from .bird_individual_id import BirdIndividualID, BirdIndividualIDSegmented
from .bristol_gorillas_2020 import BristolGorillas2020
from .brown_bear_heads import BrownBearHeads
from .cat_individual_images import CatIndividualImages
from .cattle_muzzle import CattleMuzzle
from .chicks4free_id import Chicks4FreeID
from .CHIRP import CHIRP
from .cobra_re_identification_youngstock import CoBRAReIdentificationYoungstock
from .cow_dataset import CowDataset
from .cows import Cows2021, Cows2021v2
from .ctai import CTai
from .czechlynx import CzechLynx, CzechLynxv2
from .czoo import CZoo
from .datasets import DatasetFactory, WildlifeDataset
from .dog_face_net import DogFaceNet
from .downloads import DownloadHuggingFace, DownloadINaturalist, DownloadKaggle, DownloadPrivate, DownloadURL
from .drosophila import Drosophila
from .elpephants import ELPephants
from .fire_salamanders import FireSalamanders
from .friesian_cattle import FriesianCattle2015, FriesianCattle2015v2, FriesianCattle2017
from .general import Dataset_Folder, Dataset_Metadata
from .giraffe_zebra_id import GiraffeZebraID
from .giraffes import Giraffes
from .happy_whale import HappyWhale
from .holstein_cattle_recognition import HolsteinCattleRecognition
from .hula_painted_frogs import HulaPaintedFrogs
from .humpback_whale_id import HumpbackWhaleID
from .hyena_id import HyenaID2022
from .ipanda import IPanda50
from .leopard_id import LeopardID2022
from .lion_data import LionData
from .macaque_faces import MacaqueFaces
from .melops import Melops
from .mpdd import MPDD
from .multi_cam_cows import MultiCamCows2024
from .ndd import NDD20, NDD20v2
from .newts_kent import NewtsKent
from .noaa_right_whale import NOAARightWhale
from .nyala_data import NyalaData
from .open_cows import OpenCows2020
from .polar_bear_vid_id import PolarBearVidID
from .prim_face import PrimFace
from .red_bee_reid import RedBeeReID
from .reunion_turtles import ReunionTurtles
from .rotwild_id_faces import RotwildID_Faces
from .sea_star_reid import SeaStarReID2023
from .sea_turtle_id import SeaTurtleID2022, SeaTurtleIDHeads
from .seal_id import SealID, SealIDSegmented
from .smalst import SMALST
from .southern_province_turtles import SouthernProvinceTurtles
from .spotted import LeopardID102, SpottedHyenaID109, SpottedHyenaID415
from .stripe_spotter import StripeSpotter
from .turtles_of_smsrc import TurtlesOfSMSRC
from .turtlewatch_egypt import TurtlewatchEgypt_Citizen, TurtlewatchEgypt_Master, TurtlewatchEgypt_New
from .utils import get_image, load_image
from .whaleshark_id import WhaleSharkID
from .wild_raptor_id import WildRaptorID
from .wildlife_reid_10k import WildlifeReID10k
from .zakynthos_turtles import ZakynthosTurtles
from .zindi_turtle_recall import ZindiTurtleRecall

names_all = [
    AAUZebraFish,
    AerialCattle2017,
    AmvrakikosTurtles,
    ATRW,
    BalearicLizard,
    BalearicLizardSegmented,
    BelugaIDv2,
    BirdIndividualID,
    BirdIndividualIDSegmented,
    BristolGorillas2020,
    BrownBearHeads,
    CattleMuzzle,
    CatIndividualImages,
    CHIRP,
    Chicks4FreeID,
    CoBRAReIdentificationYoungstock,
    CowDataset,
    Cows2021v2,
    CzechLynxv2,
    CTai,
    CZoo,
    DogFaceNet,
    Drosophila,
    ELPephants,
    FireSalamanders,
    FriesianCattle2015v2,
    FriesianCattle2017,
    GiraffeZebraID,
    Giraffes,
    HappyWhale,
    HolsteinCattleRecognition,
    HulaPaintedFrogs,
    HumpbackWhaleID,
    HyenaID2022,
    IPanda50,
    LeopardID102,
    LeopardID2022,
    LionData,
    MacaqueFaces,
    Melops,
    MPDD,
    MultiCamCows2024,
    NDD20v2,
    NewtsKent,
    NOAARightWhale,
    NyalaData,
    OpenCows2020,
    PolarBearVidID,
    PrimFace,
    RedBeeReID,
    ReunionTurtles,
    RotwildID_Faces,
    SealID,
    SealIDSegmented,
    SeaStarReID2023,
    SeaTurtleID2022,
    SeaTurtleIDHeads,
    SMALST,
    SouthernProvinceTurtles,
    SpottedHyenaID109,
    SpottedHyenaID415,
    StripeSpotter,
    TurtlesOfSMSRC,
    WildRaptorID,
    WhaleSharkID,
    ZakynthosTurtles,
    ZindiTurtleRecall,
]

names_wild = [
    AmvrakikosTurtles,
    BalearicLizard,
    BelugaIDv2,
    BrownBearHeads,
    CHIRP,
    CzechLynxv2,
    ELPephants,
    GiraffeZebraID,
    HappyWhale,
    HulaPaintedFrogs,
    HumpbackWhaleID,
    HyenaID2022,
    LeopardID102,
    LeopardID2022,
    Melops,
    NDD20v2,
    NOAARightWhale,
    NyalaData,
    ReunionTurtles,
    RotwildID_Faces,
    SealID,
    SeaTurtleID2022,
    SouthernProvinceTurtles,
    SpottedHyenaID109,
    SpottedHyenaID415,
    StripeSpotter,
    TurtlesOfSMSRC,
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
    LeopardID102,
    MacaqueFaces,
    MPDD,
    NyalaData,
    PolarBearVidID,
    ReunionTurtles,
    RotwildID_Faces,
    SeaTurtleIDHeads,
    SouthernProvinceTurtles,
    SpottedHyenaID109,
    StripeSpotter,
    ZakynthosTurtles,
]

names_birds = [BirdIndividualID, CHIRP]

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

names_hyenas = [
    HyenaID2022,
    SpottedHyenaID109,
    SpottedHyenaID415,
]

names_leopards = [
    LeopardID102,
    LeopardID2022,
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
    TurtlesOfSMSRC,
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

names_insect = [
    Drosophila,
    RedBeeReID,
]

names_fish = [
    AAUZebraFish,
    Melops,
]
