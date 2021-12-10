from torch.functional import broadcast_shapes
from torch.nn import ConvTranspose2d
from torch.nn import BatchNorm2d
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import LeakyReLU
from torch.nn import ReLU
from torch.nn import Tanh
from torch.nn import Sigmoid
from torch import flatten
from torch import nn


class Generator(nn.Module):
    def __init__(self, inputDim: int = 100, outputChannels: int = 1):
        super(Generator, self).__init__()

        # First set of convolutional transpose => ReLU => BN
        self.ct1 = ConvTranspose2d(
            in_channels=inputDim,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=0,
            bias=False,
        )
        self.relu1 = ReLU()
        self.batchnorm1 = BatchNorm2d(128)

        # Second set of convolutional transpose => ReLU => BN
        self.ct2 = ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.relu2 = ReLU()
        self.batchnorm2 = BatchNorm2d(64)

        # Third set of convolutional transpose => ReLU => BN
        self.ct3 = ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.relu3 = ReLU()
        self.batchnorm3 = BatchNorm2d(32)

        # Apply another upsample and transposed convolution, but
        # this time output the Tanh activation
        self.ct4 = ConvTranspose2d(
            in_channels=32,
            out_channels=outputChannels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.tanh = Tanh()

    def forward(self, x):
        # Pass the input through our first set of CONVT => RELU => BN layers
        x = self.ct1(x)
        x = self.relu1(x)
        x = self.batchnorm1(x)

        # Pass the output from the previous layer through our second
        # CONVT => RELU => BN layer set
        x = self.ct2(x)
        x = self.relu2(x)
        x = self.batchnorm2(x)

        # Pass the output from the previous layer through our last set of
        # CONVT => RELU => BN layers
        x = self.ct3(x)
        x = self.relu3(x)
        x = self.batchnorm3(x)

        # Pass the output from the previous layer through CONVT2D => TANH
        # layers to get our output
        x = self.ct4(x)
        output = self.tanh(x)

        #  Return the output
        return output


class Discriminator(nn.Module):
    def __init__(self, depth, alpha=0.2):
        super(Discriminator, self).__init__()

        # First set of CONV => RELU layers
        self.conv1 = Conv2d(
            in_channels=depth, out_channels=32, kernel_size=4, stride=2, padding=1
        )
        self.leaky_relu1 = LeakyReLU(alpha, inplace=True)

        # Second set of CONV => RELU layers
        self.conv2 = Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1
        )
        self.leaky_relu2 = LeakyReLU(alpha, inplace=True)

        # First (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=3136, out_features=512)
        self.leaky_relu3 = LeakyReLU(alpha, inplace=True)

        # Sigmoid layer outputting a single value
        self.fc2 = Linear(in_features=512, out_features=1)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        # pass the input through first set of CONV => RELU layers
        x = self.conv1(x)
        x = self.leaky_relu1(x)

        # pass the output from the previous layer through our second
        # set of CONV => RELU layers
        x = self.conv2(x)
        x = self.leaky_relu2(x)

        # flatten the output from the previous layer and pass it
        # through our first (and only) set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.leaky_relu3(x)

        # pass the output from the previous layer through our sigmoid
        # layer outputting a single value
        x = self.fc2(x)
        output = self.sigmoid(x)

        # return the output
        return output
