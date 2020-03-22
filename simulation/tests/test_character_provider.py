from unittest import TestCase

from simulation import CharacterRenderer


class TestCharacterProvider(TestCase):
    def test_prerender_all(self):
        CharacterRenderer().prerender_all()
