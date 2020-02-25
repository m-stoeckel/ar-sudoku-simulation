from pathlib import Path
from typing import Union

import numpy as np

from sudoku.render.digital_composition import AlphaComposition
from sudoku.render.layers import DigitalCompositionLayer, DrawingLayer, SubstrateLayer


class LayeredPaperRenderer:
    def __init__(self, substrate_layer: SubstrateLayer, shape=(1000, 1000), print_area=0.98):
        self.shape = shape
        self.print_area = np.clip(print_area, 0.0, 1.0)

        self.backside_drawing_layer = DrawingLayer(shape, backside=True)
        self.backside_print_layer = DigitalCompositionLayer(shape, backside=True)
        self.background_layer = substrate_layer
        self.print_layer = DigitalCompositionLayer(shape)
        self.drawing_layer = DrawingLayer(shape)

    def render(self):
        draw_composition = AlphaComposition()
        print_composition = AlphaComposition(self.print_area)

        result = self.background_layer.compose()

        if self.backside_drawing_layer.not_empty():
            result = print_composition(self.backside_drawing_layer.compose(), result)
        if self.backside_print_layer.not_empty():
            result = draw_composition(self.backside_print_layer.compose(), result)

        if self.print_layer.not_empty():
            result = print_composition(result, self.print_layer.compose())
        if self.drawing_layer.not_empty():
            result = draw_composition(result, self.drawing_layer.compose())

        return result


class MeshedLayeredPaperRenderer(LayeredPaperRenderer):
    def __init__(self, mesh=None, mesh_file: Union[str, Path] = None, **kwargs):
        super().__init__(**kwargs)
        if mesh is not None:
            self.mesh = mesh
        elif mesh_file is not None:
            self.mesh = self.load_mesh(Path(mesh_file))
        else:
            raise ValueError("Either a mesh or a path to a valid mash file must be passed!")

    def load_mesh(self, mesh_file: Path):
        # TODO: Implement mesh loading
        raise NotImplementedError
