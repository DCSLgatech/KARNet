bnm = 0.1

layers_encoder_256_64 = [

    {"type": "Conv2d", "in_channels": 1, "out_channels": 2, "kernel_size": 3, "stride": 2, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 2, "momentum": bnm},
    {"type": "ReLU"},

    {"type": "Conv2d", "in_channels": 2, "out_channels": 4, "kernel_size": 3, "stride": 2, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 4, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 4, "out_channels": 4, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 4, "momentum": bnm},
    {"type": "ReLU"},

    {"type": "Conv2d", "in_channels": 4, "out_channels": 8, "kernel_size": 3, "stride": 2, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 8, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 8, "out_channels": 8, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 8, "momentum": bnm},
    {"type": "ReLU"},

    {"type": "Conv2d", "in_channels": 8, "out_channels": 16, "kernel_size": 3, "stride": 2, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 16, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 16, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 16, "momentum": bnm},
    {"type": "ReLU"},

    {"type": "Conv2d", "in_channels": 16, "out_channels": 32, "kernel_size": 3, "stride": 2, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 32, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 32, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 32, "momentum": bnm},
    {"type": "ReLU"},

    {"type": "Conv2d", "in_channels": 32, "out_channels": 48, "kernel_size": 3, "stride": 2, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 48, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 48, "out_channels": 64, "kernel_size": 3, "stride": 2, "padding": 0,},
    {"type": "BatchNorm2d", "num_features": 64, "momentum": bnm},
    {"type": "ReLU"},
]

layers_decoder_256_64 = [

    {"type": "ConvTranspose2d", "in_channels": 64, "out_channels": 48, "kernel_size": 3, "stride": 2, "padding": 0, "output_padding": 1},
    {"type": "BatchNorm2d", "num_features": 48, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "ConvTranspose2d", "in_channels": 48, "out_channels": 32, "kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
    {"type": "BatchNorm2d", "num_features": 32, "momentum": bnm},
    {"type": "ReLU"},

    {"type": "Conv2d", "in_channels": 32, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 32, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "ConvTranspose2d", "in_channels": 32, "out_channels": 16, "kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
    {"type": "BatchNorm2d", "num_features": 16, "momentum": bnm},
    {"type": "ReLU"},

    {"type": "Conv2d", "in_channels": 16, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 16, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "ConvTranspose2d", "in_channels": 16, "out_channels": 8, "kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
    {"type": "BatchNorm2d", "num_features": 8, "momentum": bnm},
    {"type": "ReLU"},

    {"type": "Conv2d", "in_channels": 8, "out_channels": 8, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 8, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "ConvTranspose2d", "in_channels": 8, "out_channels": 4, "kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
    {"type": "BatchNorm2d", "num_features": 4, "momentum": bnm},
    {"type": "ReLU"},

    {"type": "Conv2d", "in_channels": 4, "out_channels": 4, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 4, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "ConvTranspose2d", "in_channels": 4, "out_channels": 2, "kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
    {"type": "BatchNorm2d", "num_features": 2, "momentum": bnm},
    {"type": "ReLU"},

    {"type": "ConvTranspose2d", "in_channels": 2, "out_channels": 1, "kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
    {"type": "BatchNorm2d", "num_features": 1, "momentum": bnm},
    {"type": "ReLU"},
]

layers_encoder_256_128 = [

    {"type": "Conv2d", "in_channels": 1, "out_channels": 2, "kernel_size": 3, "stride": 2, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 2, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 2, "out_channels": 2, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 2, "momentum": bnm},
    {"type": "ReLU"},

    {"type": "Conv2d", "in_channels": 2, "out_channels": 4, "kernel_size": 3, "stride": 2, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 4, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 4, "out_channels": 4, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 4, "momentum": bnm},
    {"type": "ReLU"},

    {"type": "Conv2d", "in_channels": 4, "out_channels": 8, "kernel_size": 3, "stride": 2, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 8, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 8, "out_channels": 8, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 8, "momentum": bnm},
    {"type": "ReLU"},

    {"type": "Conv2d", "in_channels": 8, "out_channels": 16, "kernel_size": 3, "stride": 2, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 16, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 16, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 16, "momentum": bnm},
    {"type": "ReLU"},

    {"type": "Conv2d", "in_channels": 16, "out_channels": 32, "kernel_size": 3, "stride": 2, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 32, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 32, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 32, "momentum": bnm},
    {"type": "ReLU"},

    {"type": "Conv2d", "in_channels": 32, "out_channels": 64, "kernel_size": 3, "stride": 2, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 64, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 64, "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 64, "momentum": bnm},
    {"type": "ReLU"},

    {"type": "Conv2d", "in_channels": 64, "out_channels": 128, "kernel_size": 3, "stride": 2, "padding": 0,},
    {"type": "BatchNorm2d", "num_features": 128, "momentum": bnm},
    {"type": "ReLU"},
]

layers_decoder_256_128 = [
    # 4x4x256
    {"type": "ConvTranspose2d", "in_channels": 128, "out_channels": 64, "kernel_size": 3, "stride": 2, "padding": 0, "output_padding": 1},
    {"type": "BatchNorm2d", "num_features": 64, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 64, "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 64, "momentum": bnm},
    {"type": "ReLU"},
    # 8x8x128
    {"type": "ConvTranspose2d", "in_channels": 64, "out_channels": 32, "kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
    {"type": "BatchNorm2d", "num_features": 32, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 32, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 32, "momentum": bnm},
    {"type": "ReLU"},
    # 16x16x64
    {"type": "ConvTranspose2d", "in_channels": 32, "out_channels": 16, "kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
    {"type": "BatchNorm2d", "num_features": 16, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 16, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 16, "momentum": bnm},
    {"type": "ReLU"},
    # 32x32x32
    {"type": "ConvTranspose2d", "in_channels": 16, "out_channels": 8, "kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
    {"type": "BatchNorm2d", "num_features": 8, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 8, "out_channels": 8, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 8, "momentum": bnm},
    {"type": "ReLU"},
    # 64x64x16
    {"type": "ConvTranspose2d", "in_channels": 8, "out_channels": 4, "kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
    {"type": "BatchNorm2d", "num_features": 4, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 4, "out_channels": 4, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 4, "momentum": bnm},
    {"type": "ReLU"},
    # 128x128x8
    {"type": "ConvTranspose2d", "in_channels": 4, "out_channels": 2, "kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
    {"type": "BatchNorm2d", "num_features": 2, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 2, "out_channels": 2, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 2, "momentum": bnm},
    {"type": "ReLU"},
    # 256x256x3
    {"type": "ConvTranspose2d", "in_channels": 2, "out_channels": 1, "kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
    {"type": "BatchNorm2d", "num_features": 1, "momentum": bnm},
    {"type": "ReLU"},
]

layers_encoder_256_128_c = [

    {"type": "Conv2d", "in_channels": 2, "out_channels": 2, "kernel_size": 3, "stride": 2, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 2, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 2, "out_channels": 2, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 2, "momentum": bnm},
    {"type": "ReLU"},

    {"type": "Conv2d", "in_channels": 2, "out_channels": 4, "kernel_size": 3, "stride": 2, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 4, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 4, "out_channels": 4, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 4, "momentum": bnm},
    {"type": "ReLU"},

    {"type": "Conv2d", "in_channels": 4, "out_channels": 8, "kernel_size": 3, "stride": 2, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 8, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 8, "out_channels": 8, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 8, "momentum": bnm},
    {"type": "ReLU"},

    {"type": "Conv2d", "in_channels": 8, "out_channels": 16, "kernel_size": 3, "stride": 2, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 16, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 16, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 16, "momentum": bnm},
    {"type": "ReLU"},

    {"type": "Conv2d", "in_channels": 16, "out_channels": 32, "kernel_size": 3, "stride": 2, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 32, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 32, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 32, "momentum": bnm},
    {"type": "ReLU"},

    {"type": "Conv2d", "in_channels": 32, "out_channels": 64, "kernel_size": 3, "stride": 2, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 64, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 64, "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 64, "momentum": bnm},
    {"type": "ReLU"},

    {"type": "Conv2d", "in_channels": 64, "out_channels": 128, "kernel_size": 3, "stride": 2, "padding": 0,},
    {"type": "BatchNorm2d", "num_features": 128, "momentum": bnm},
    {"type": "ReLU"},
]

# New
layers_encoder_256_256 = [
    {"type": "Conv2d", "in_channels": 1, "out_channels": 4, "kernel_size": 3, "stride": 2, "padding": 1,},
    # 128x128x8
    {"type": "BatchNorm2d", "num_features": 4, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 4, "out_channels": 4, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 4, "momentum": bnm},
    {"type": "ReLU"},
    # 64x64x16
    {"type": "Conv2d", "in_channels": 4, "out_channels": 8, "kernel_size": 3, "stride": 2, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 8, "momentum": bnm},
    {"type": "Conv2d", "in_channels": 8, "out_channels": 8, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "ReLU"},
    {"type": "BatchNorm2d", "num_features": 8, "momentum": bnm},
    {"type": "ReLU"},
    # 32x32x32
    {"type": "Conv2d", "in_channels": 8, "out_channels": 16, "kernel_size": 3, "stride": 2, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 16, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 16, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 16, "momentum": bnm},
    {"type": "ReLU"},
    # 16x16x64
    {"type": "Conv2d", "in_channels": 16, "out_channels": 32, "kernel_size": 3, "stride": 2, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 32, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 32, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 32, "momentum": bnm},
    {"type": "ReLU"},
    # 8x8x128
    {"type": "Conv2d", "in_channels": 32, "out_channels": 64, "kernel_size": 3, "stride": 2, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 64, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 64, "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 64, "momentum": bnm},
    {"type": "ReLU"},
    # 4x4x256
    {"type": "Conv2d", "in_channels": 64, "out_channels": 128, "kernel_size": 3, "stride": 2, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 128, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 128, "out_channels": 128, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 128, "momentum": bnm},
    {"type": "ReLU"},
    # 1x1x512
    {"type": "Conv2d", "in_channels": 128, "out_channels": 256, "kernel_size": 3, "stride": 2, "padding": 0,},
    {"type": "BatchNorm2d", "num_features": 256, "momentum": bnm},
    {"type": "ReLU"},
]
layers_decoder_256_256 = [
    # 4x4x256
    {"type": "ConvTranspose2d", "in_channels": 256, "out_channels": 128, "kernel_size": 3, "stride": 2, "padding": 0, "output_padding": 1},
    {"type": "BatchNorm2d", "num_features": 128, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 128, "out_channels": 128, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 128, "momentum": bnm},
    {"type": "ReLU"},
    # 8x8x128
    {"type": "ConvTranspose2d", "in_channels": 128, "out_channels": 64, "kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
    {"type": "BatchNorm2d", "num_features": 64, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "BatchNorm2d", "num_features": 64, "momentum": bnm},
    {"type": "Conv2d", "in_channels": 64, "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "ReLU"},
    # 16x16x64
    {"type": "ConvTranspose2d", "in_channels": 64, "out_channels": 32, "kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
    {"type": "BatchNorm2d", "num_features": 32, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 32, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 32, "momentum": bnm},
    {"type": "ReLU"},
    # 32x32x32
    {"type": "ConvTranspose2d", "in_channels": 32, "out_channels": 16, "kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
    {"type": "BatchNorm2d", "num_features": 16, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 16, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 16, "momentum": bnm},
    {"type": "ReLU"},
    # 64x64x16
    {"type": "ConvTranspose2d", "in_channels": 16, "out_channels": 8, "kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
    {"type": "BatchNorm2d", "num_features": 8, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 8, "out_channels": 8, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 8, "momentum": bnm},
    {"type": "ReLU"},
    # 128x128x8
    {"type": "ConvTranspose2d", "in_channels":8, "out_channels": 4, "kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
    {"type": "BatchNorm2d", "num_features": 4, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 4, "out_channels": 4, "kernel_size": 3, "stride": 1, "padding": 1,},
    {"type": "BatchNorm2d", "num_features": 4, "momentum": bnm},
    {"type": "ReLU"},
    # 256x256x3
    {"type": "ConvTranspose2d", "in_channels": 4, "out_channels": 1, "kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
    {"type": "BatchNorm2d", "num_features": 1, "momentum": bnm},
    {"type": "ReLU"},
]

layers_resnet_decoder = [
    # 3x3x256
    {"type": "Conv2d", "in_channels": 256, "out_channels": 256, "kernel_size": 3, "stride": 1, "padding": 1},
    {"type": "BatchNorm2d", "num_features": 256, "momentum": bnm},
    {"type": "ReLU"},
    # 7x7x128
    {"type": "ConvTranspose2d", "in_channels": 256, "out_channels": 128, "kernel_size": 3, "stride": 2, "padding": 0, "output_padding": 0},
    {"type": "BatchNorm2d", "num_features": 128, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 128, "out_channels": 128, "kernel_size": 3, "stride": 1, "padding": 1},
    {"type": "BatchNorm2d", "num_features": 128, "momentum": bnm},
    {"type": "ReLU"},
    # 14x14x64
    {"type": "ConvTranspose2d", "in_channels": 128, "out_channels": 64, "kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
    {"type": "BatchNorm2d", "num_features": 64, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 64, "out_channels": 64, "kernel_size": 3, "stride": 1, "padding": 1},
    {"type": "BatchNorm2d", "num_features": 64, "momentum": bnm},
    {"type": "ReLU"},
    # 28x28x32
    {"type": "ConvTranspose2d", "in_channels": 64, "out_channels": 32, "kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
    {"type": "BatchNorm2d", "num_features": 32, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 32, "out_channels": 32, "kernel_size": 3, "stride": 1, "padding": 1},
    {"type": "BatchNorm2d", "num_features": 32, "momentum": bnm},
    {"type": "ReLU"},
    # 56x56x16
    {"type": "ConvTranspose2d", "in_channels": 32, "out_channels": 16, "kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
    {"type": "BatchNorm2d", "num_features": 16, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 16, "out_channels": 16, "kernel_size": 3, "stride": 1, "padding": 1},
    {"type": "BatchNorm2d", "num_features": 16, "momentum": bnm},
    {"type": "ReLU"},
    # 112x112x8
    {"type": "ConvTranspose2d", "in_channels": 16, "out_channels": 8, "kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
    {"type": "BatchNorm2d", "num_features": 8, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 8, "out_channels": 8, "kernel_size": 3, "stride": 1, "padding": 1},
    {"type": "BatchNorm2d", "num_features": 8, "momentum": bnm},
    {"type": "ReLU"},
    # 224x224x3
    {"type": "ConvTranspose2d", "in_channels": 8, "out_channels": 3, "kernel_size": 3, "stride": 2, "padding": 1, "output_padding": 1},
    {"type": "BatchNorm2d", "num_features": 3, "momentum": bnm},
    {"type": "ReLU"},
    {"type": "Conv2d", "in_channels": 3, "out_channels": 3, "kernel_size": 3, "stride": 1, "padding": 1},
    {"type": "BatchNorm2d", "num_features": 8, "momentum": bnm},
    {"type": "Sigmoid"},
]
