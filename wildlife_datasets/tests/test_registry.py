import unittest

from wildlife_datasets import datasets

SPECIES_GROUPS = {
    "names_birds": ["birds"],
    "names_cows": ["cows"],
    "names_dogs": ["dogs"],
    "names_fish": ["fish"],
    "names_giraffes_zebras": ["giraffes", "zebras", "giraffes+zebras"],
    "names_insect": ["flies", "bees"],
    "names_hyenas": ["hyenas"],
    "names_leopards": ["leopards"],
    "names_primates": ["monkeys", "macaques", "chimpanzees", "gorillas"],
    "names_turtles": ["sea turtles"],
    "names_whales_dolphins": ["whales", "dolphins+whales", "dolphins"],
}


class TestNamesAll(unittest.TestCase):
    def test_every_member_has_non_empty_summary(self):
        for cls in datasets.names_all:
            self.assertTrue(bool(cls.summary), f"{cls.__name__} has empty summary")

    def test_no_member_is_outdated(self):
        for cls in datasets.names_all:
            self.assertFalse(cls.outdated_dataset, f"{cls.__name__} is outdated")

    def test_every_member_is_a_wildlife_dataset_subclass(self):
        for cls in datasets.names_all:
            self.assertTrue(issubclass(cls, datasets.WildlifeDataset))

    def test_no_duplicates(self):
        self.assertEqual(len(datasets.names_all), len(set(datasets.names_all)))

    def test_meta_datasets_are_currently_included(self):
        self.assertIn(datasets.AnimalCLEF2025, datasets.names_all)
        self.assertIn(datasets.AnimalCLEF2026, datasets.names_all)
        self.assertIn(datasets.WildlifeReID10k, datasets.names_all)


class TestNamesWild(unittest.TestCase):
    def test_subset_of_names_all(self):
        self.assertTrue(set(datasets.names_wild).issubset(set(datasets.names_all)))

    def test_every_member_is_tagged_wild(self):
        for cls in datasets.names_wild:
            self.assertIs(cls.summary.get("wild"), True, f"{cls.__name__} is not tagged wild")


class TestNamesSmall(unittest.TestCase):
    def test_subset_of_names_all(self):
        self.assertTrue(set(datasets.names_small).issubset(set(datasets.names_all)))

    def test_every_member_is_within_size_threshold(self):
        for cls in datasets.names_small:
            size = cls.summary.get("size")
            self.assertIsNotNone(size, f"{cls.__name__} has no size")
            self.assertLessEqual(size, 1000, f"{cls.__name__} exceeds the size threshold")


if __name__ == "__main__":
    unittest.main()
