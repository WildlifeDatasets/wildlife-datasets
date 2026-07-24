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


def _all_dataset_classes():
    seen = set()
    stack = list(WildlifeDataset.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls not in seen:
            seen.add(cls)
            stack.extend(cls.__subclasses__())
    return sorted(seen, key=lambda c: c.__name__)


def _subset_name(classes, animal_names):
    return [c for c in classes if c.summary.get("animals_simple") in animal_names]


def _subset_small(names_all, max_size=1000):
    subset = []
    for c in names_all:
        size = c.summary.get("size")
        if size and size <= max_size:
            subset.append(c)
    return subset


def _subset_wild(names_all):
    return [c for c in names_all if c.summary.get("wild") is True]


_classes = _all_dataset_classes()

names_all = [c for c in _classes if bool(c.summary) and not c.outdated_dataset]
names_small = _subset_small(names_all)
names_wild = _subset_wild(names_all)

names_birds = _subset_name(names_all, ["birds"])
names_cows = _subset_name(names_all, ["cows"])
names_dogs = _subset_name(names_all, ["dogs"])
names_fish = _subset_name(names_all, ["fish"])
names_giraffes_zebras = _subset_name(names_all, ["giraffes", "zebras", "giraffes+zebras"])
names_insect = _subset_name(names_all, ["flies", "bees"])
names_hyenas = _subset_name(names_all, ["hyenas"])
names_leopards = _subset_name(names_all, ["leopards"])
names_primates = _subset_name(names_all, ["monkeys", "macaques", "chimpanzees", "gorillas"])
names_turtles = _subset_name(names_all, ["sea turtles"])
names_whales_dolphins = _subset_name(names_all, ["whales", "dolphins+whales", "dolphins"])
