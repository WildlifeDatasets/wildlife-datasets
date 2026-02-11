import os
import numpy as np
import pandas as pd
from PIL import Image
from . import utils
from .datasets import WildlifeDataset
from .downloads import DownloadURL

summary = {
    'licenses': 'Other',
    'licenses_url': 'https://visiome.neuroinf.jp/primface/',
    'url': 'https://visiome.neuroinf.jp/primface/',
    'publication_url': None,
    'cite': 'primface',
    'animals': {'rhesus monkey', 'japanese monkey', 'chimpanzee'},
    'animals_simple': 'monkeys',
    'real_animals': True,
    'year': 2012,
    'reported_n_total': 1282,
    'reported_n_individuals': 68,
    'wild': False,
    'clear_photos': True,
    'pose': 'single',
    'unique_pattern': False,
    'from_video': False,
    'cropped': True,
    'span': 'short',
    'size': 3700,
}

class PrimFace(DownloadURL, WildlifeDataset):
    summary = summary
    downloads = [
        ('https://visiome.neuroinf.jp/database/file/1508/j01.zip', 'j01.zip'),
        ('https://visiome.neuroinf.jp/database/file/1518/j02.zip', 'j02.zip'),
        ('https://visiome.neuroinf.jp/database/file/1525/j03.zip', 'j03.zip'),
        ('https://visiome.neuroinf.jp/database/file/1530/j04.zip', 'j04.zip'),
        ('https://visiome.neuroinf.jp/database/file/1539/j05.zip', 'j05.zip'),
        ('https://visiome.neuroinf.jp/database/file/1504/j06.zip', 'j06.zip'),
        ('https://visiome.neuroinf.jp/database/file/1513/j07.zip', 'j07.zip'),
        ('https://visiome.neuroinf.jp/database/file/1553/j08.zip', 'j08.zip'),
        ('https://visiome.neuroinf.jp/database/file/1612/j09.zip', 'j09.zip'),
        ('https://visiome.neuroinf.jp/database/file/1555/j10.zip', 'j10.zip'),
        ('https://visiome.neuroinf.jp/database/file/1557/j11.zip', 'j11.zip'),
        ('https://visiome.neuroinf.jp/database/file/1559/j12.zip', 'j12.zip'),
        ('https://visiome.neuroinf.jp/database/file/1561/j13.zip', 'j13.zip'),
        ('https://visiome.neuroinf.jp/database/file/1516/j14.zip', 'j14.zip'),
        ('https://visiome.neuroinf.jp/database/file/1522/j15.zip', 'j15.zip'),
        ('https://visiome.neuroinf.jp/database/file/1526/j16.zip', 'j16.zip'),
        ('https://visiome.neuroinf.jp/database/file/1531/j17.zip', 'j17.zip'),
        ('https://visiome.neuroinf.jp/database/file/1535/j18.zip', 'j18.zip'),
        ('https://visiome.neuroinf.jp/database/file/1573/j19.2.zip', 'j19.2.zip'),
        ('https://visiome.neuroinf.jp/database/file/1503/r01.zip', 'r01.zip'),
        ('https://visiome.neuroinf.jp/database/file/1509/r02.zip', 'r02.zip'),
        ('https://visiome.neuroinf.jp/database/file/1515/r03.zip', 'r03.zip'),
        ('https://visiome.neuroinf.jp/database/file/1521/r04.zip', 'r04.zip'),
        ('https://visiome.neuroinf.jp/database/file/1615/r05.zip', 'r05.zip'),
        ('https://visiome.neuroinf.jp/database/file/1617/r06.zip', 'r06.zip'),
        ('https://visiome.neuroinf.jp/database/file/1619/r07.zip', 'r07.zip'),
        ('https://visiome.neuroinf.jp/database/file/1621/r08.zip', 'r08.zip'),
        ('https://visiome.neuroinf.jp/database/file/1623/r09.zip', 'r09.zip'),
        ('https://visiome.neuroinf.jp/database/file/1627/r10.zip', 'r10.zip'),
        ('https://visiome.neuroinf.jp/database/file/1629/r11.zip', 'r11.zip'),
        ('https://visiome.neuroinf.jp/database/file/1633/r12.zip', 'r12.zip'),
        ('https://visiome.neuroinf.jp/database/file/1635/r13.zip', 'r13.zip'),
        ('https://visiome.neuroinf.jp/database/file/1637/r14.zip', 'r14.zip'),
        ('https://visiome.neuroinf.jp/database/file/1639/r15.zip', 'r15.zip'),
        ('https://visiome.neuroinf.jp/database/file/1641/r16.zip', 'r16.zip'),
        ('https://visiome.neuroinf.jp/database/file/1643/r17.zip', 'r17.zip'),
        ('https://visiome.neuroinf.jp/database/file/1645/r18.zip', 'r18.zip'),
        ('https://visiome.neuroinf.jp/database/file/1647/r19.zip', 'r19.zip'),
        ('https://visiome.neuroinf.jp/database/file/1649/r20.zip', 'r20.zip'),
        ('https://visiome.neuroinf.jp/database/file/1651/r21.zip', 'r21.zip'),
        ('https://visiome.neuroinf.jp/database/file/1653/r22.zip', 'r22.zip'),
        ('https://visiome.neuroinf.jp/database/file/1655/r23.zip', 'r23.zip'),
        ('https://visiome.neuroinf.jp/database/file/1657/r24.zip', 'r24.zip'),
        ('https://visiome.neuroinf.jp/database/file/1659/r25.zip', 'r25.zip'),
        ('https://visiome.neuroinf.jp/database/file/1661/r26.zip', 'r26.zip'),
        ('https://visiome.neuroinf.jp/database/file/1663/r27.zip', 'r27.zip'),
        ('https://visiome.neuroinf.jp/database/file/1665/r28.zip', 'r28.zip'),
        ('https://visiome.neuroinf.jp/database/file/1667/r29.zip', 'r29.zip'),
        ('https://visiome.neuroinf.jp/database/file/1669/r30.zip', 'r30.zip'),
        ('https://visiome.neuroinf.jp/database/file/1671/r31.zip', 'r31.zip'),
        ('https://visiome.neuroinf.jp/database/file/1673/r32.zip', 'r32.zip'),
        ('https://visiome.neuroinf.jp/database/file/1675/r33.zip', 'r33.zip'),
        ('https://visiome.neuroinf.jp/database/file/1677/r34.zip', 'r34.zip'),
        ('https://visiome.neuroinf.jp/database/file/1679/r35.zip', 'r35.zip'),
        ('https://visiome.neuroinf.jp/database/file/1681/r36.zip', 'r36.zip'),
        ('https://visiome.neuroinf.jp/database/file/1683/r37.zip', 'r37.zip'),
        ('https://visiome.neuroinf.jp/database/file/1685/r38.zip', 'r38.zip'),
        ('https://visiome.neuroinf.jp/database/file/1563/c01.zip', 'c01.zip'),
        ('https://visiome.neuroinf.jp/database/file/1565/c02.zip', 'c02.zip'),
        ('https://visiome.neuroinf.jp/database/file/1567/c03.zip', 'c03.zip'),
        ('https://visiome.neuroinf.jp/database/file/1569/c04.zip', 'c04.zip'),
        ('https://visiome.neuroinf.jp/database/file/1571/c05.zip', 'c05.zip'),
        ('https://visiome.neuroinf.jp/database/file/1691/c06.zip', 'c06.zip'),
        ('https://visiome.neuroinf.jp/database/file/1693/c07.zip', 'c07.zip'),
        ('https://visiome.neuroinf.jp/database/file/1695/c08.zip', 'c08.zip'),
        ('https://visiome.neuroinf.jp/database/file/1697/c09.zip', 'c09.zip'),
        ('https://visiome.neuroinf.jp/database/file/1699/c10.zip', 'c10.zip'),
        ('https://visiome.neuroinf.jp/database/file/1701/c11.zip', 'c11.zip'),
    ]

    def create_catalogue(self) -> pd.DataFrame:
        # Find all images in root
        assert self.root is not None
        data = utils.find_images(self.root)
        
        # Finalize the dataframe
        species = {'c': 'chimpanzee', 'j': 'japanese monkey', 'r': 'rhesus monkey'}
        df = pd.DataFrame({
            'image_id': data['file'].apply(lambda x: os.path.splitext(x)[0]),
            'path': data['path'] + os.path.sep + data['file'],
            'identity': data['file'].apply(lambda x: x.split('_')[0]),
            'species': data['file'].apply(lambda x: species[x[0]]),
        })
        return self.finalize_catalogue(df)

    def load_image(self, path: str) -> Image.Image:
        """Load an image with `path`. Need to write this because PIL loads (w,h,4), where the last channel is transparency.

        Args:
            path (str): Path to the image.

        Returns:
            Loaded image.
        """

        img = Image.open(path)
        return Image.fromarray(np.array(img)[:,:,:3])
