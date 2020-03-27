from unittest import TestCase

from matplotlib import pyplot as plt

from simulation.transforms import *


class ImageTransformTests(TestCase):
    def setUp(self):
        self.digit1 = cv2.imread("../../datasets/digits/1.png", cv2.IMREAD_GRAYSCALE)
        self.digit1 = cv2.resize(self.digit1, (28, 28))
        self.digit2 = cv2.imread("../../datasets/curated/50/15.png", cv2.IMREAD_GRAYSCALE)
        self.digit2 = cv2.resize(self.digit2, (28, 28))

    def apply_transform(self, transform):
        for digit in [self.digit1, self.digit2]:
            tdigit = transform.apply(digit)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2, 1))
            fig.suptitle(transform.__class__.__name__, fontsize='8')
            ax1.axis('off')
            ax2.axis('off')
            ax1.imshow(digit, cmap="gray", interpolation='none')
            ax2.imshow(tdigit, cmap="gray", interpolation='none')
            plt.show()

    def test_with_sudoku(self):
        img = cv2.imread("../../sudoku.jpeg", cv2.IMREAD_GRAYSCALE)
        img = GaussianNoise().apply(img)
        img = RandomPerspectiveTransform(0.2).apply(img)
        img = GaussianBlur().apply(img)

        plt.imshow(img)
        plt.imshow(img, cmap="gray")
        plt.axis('off')
        plt.show()

    def test_EmbedInGrid(self):
        transform = EmbedInGrid()
        self.apply_transform(transform)

    def test_GaussianBlur(self):
        transform = GaussianBlur()
        self.apply_transform(transform)

    def test_BoxBlur(self):
        transform = BoxBlur()
        self.apply_transform(transform)

    def test_GaussianNoise(self):
        transform = GaussianNoise()
        self.apply_transform(transform)

    def test_PoissonNoise(self):
        transform = PoissonNoise()
        self.apply_transform(transform)

    def test_UniformNoise(self):
        transform = UniformNoise()
        self.apply_transform(transform)

    def test_SaltAndPepperNoise(self):
        transform = SaltAndPepperNoise()
        self.apply_transform(transform)

    def test_GrainNoise(self):
        transform = GrainNoise()
        self.apply_transform(transform)

    def test_SpeckleNoise(self):
        transform = SpeckleNoise()
        self.apply_transform(transform)

    def test_SharpenFilter(self):
        transform = SharpenFilter()
        self.apply_transform(transform)

    def test_ReliefFilter(self):
        transform = ReliefFilter()
        self.apply_transform(transform)

    def test_EdgeFilter(self):
        transform = EdgeFilter()
        self.apply_transform(transform)

    def test_UnsharpMaskingFilter(self):
        transform = UnsharpMaskingFilter3x3()
        self.apply_transform(transform)

    def test_RandomPerspectiveTransform(self):
        transform = RandomPerspectiveTransform()
        self.apply_transform(transform)

    def test_JPEGEncode(self):
        transform = JPEGEncode()
        self.apply_transform(transform)

    def test_Dilate(self):
        transform = Dilate()
        self.apply_transform(transform)

    def test_Dilate2(self):
        b_digit = self.digit1.copy()
        self.digit1 = SaltAndPepperNoise().apply(self.digit1)
        transform = Dilate()
        self.apply_transform(transform)
        self.digit1 = b_digit

    def test_RescaleIntermediateTransforms(self):
        transform = RescaleIntermediateTransforms(
            (92, 92),
            [SaltAndPepperNoise(amount=0.002, ratio=1), Dilate()],
            inter_initial=cv2.INTER_LINEAR, inter_consecutive=cv2.INTER_AREA
        )
        self.apply_transform(transform)

    def generate_composition(self):
        transform = RandomPerspectiveTransform()
        images = [[] for _ in range(9)]
        for i in range(0, 9):
            img = cv2.imread(f"../datasets/digits/{i * 917}.png", cv2.IMREAD_GRAYSCALE)
            img = cv2.rectangle(img, (16, 16), (112, 112), (255, 255, 255), 2)
            images[i].append(img)
            for _ in range(4):
                digit = transform.apply(img)
                images[i].append(digit)
        imgs = np.hstack([np.vstack(digits) for digits in images])
        imgs = cv2.bitwise_not(imgs)
        imgs.save("composition.png")
        plt.imshow(imgs, cmap="gray")
        plt.axis('off')
        plt.show()
