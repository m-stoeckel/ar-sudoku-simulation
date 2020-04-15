import numpy as np

from simulation.render import AlphaComposition
from simulation.render.layers import DigitalCompositionLayer, DrawingLayer, SubstrateLayer


class LayeredPaperRenderer:
    """
    A five-layer renderer for substrate printing simulation. This renderer considers two layers for each side of the
    substrate layer. One for printed elements and one for handwritten elements, which are drawn on top of the printed
    elements.

    Qu (2013) show that the predicted color for printing in a simple model is a linear combination of the color of the
    used ink and the color of the substrate (see Eq. (5.1)) where the color intensity of the ink is controlled by the
    *print area*, which determines how much of any given space is covered by an ink dot. These models only consider
    single ink prints meaning the final color has to be computed from all print colors considering the subtractive
    nature of ink color combination.

    This can be simplified to a pixel-based additive model for our purposes, where the print area is modeled by the
    pixels alpha value and combined colors are computed additively without forward-backward conversions to subtractive
    color combination models. Thus the simulation of printing decomposes to an *Alpha Composition* of substrate,
    printing and handwritten layers.

    :sources: 'Color Prediction and Separation Models in Printing', Yuanyuan Qu (2013)
    """

    def __init__(self, substrate_layer: SubstrateLayer, print_area=0.99):
        """

        Args:
            substrate_layer(:py:class:`SubstrateLayer <simulation.render.layers.base_layers.SubstrateLayer>`): The
                substrate (eg. paper) layer.
            print_area(float): The print area to be simulated as a normalized float. (Default value = 0.99)

        """
        self.substrate_layer = substrate_layer
        self.shape = substrate_layer.shape
        self.print_area = float(np.clip(print_area, 0.0, 1.0))

        self.backside_drawing_layer = DrawingLayer(self.shape, backside=True)
        self.backside_print_layer = DigitalCompositionLayer(self.shape, backside=True)
        self.print_layer = DigitalCompositionLayer(self.shape)
        self.drawing_layer = DrawingLayer(self.shape)

    def render(self):
        draw_composition = AlphaComposition()
        print_composition = AlphaComposition(self.print_area)

        result = self.substrate_layer.compose()

        if self.backside_drawing_layer.not_empty():
            result = print_composition(self.backside_drawing_layer.compose(), result)
        if self.backside_print_layer.not_empty():
            result = draw_composition(self.backside_print_layer.compose(), result)

        if self.print_layer.not_empty():
            result = print_composition(result, self.print_layer.compose())
        if self.drawing_layer.not_empty():
            result = draw_composition(result, self.drawing_layer.compose())

        return result

# class MeshedLayeredPaperRenderer(LayeredPaperRenderer):
#     def __init__(self, mesh=None, mesh_file: Union[str, Path] = None, **kwargs):
#         super().__init__(**kwargs)
#         if mesh is not None:
#             self.mesh = mesh
#         elif mesh_file is not None:
#             self.mesh = self.load_mesh(Path(mesh_file))
#         else:
#             raise ValueError("Either a mesh or a path to a valid mash file must be passed!")
#
#     def load_mesh(self, mesh_file: Path):
#         # TODO: Implement mesh loading
#         raise NotImplementedError
