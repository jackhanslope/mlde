from ml_downscaling_emulator.score_sde_pytorch_hja22.configs import \
    default_xarray_configs


def get_config():
    config = default_xarray_configs.get_default_configs()

    # training
    training = config.training
    training.sde = "subvpsde"
    training.continuous = True
    training.reduce_mean = True

    # data
    data = config.data
    data.dataset = "hurricanes"
    data.dataset_name = "hurricanes"
    data.image_size = 32
    data.image_size_x = 32
    data.image_size_y = 56
    # This will be input_data_steps * $(number of channels) when the dataset was created
    # in the hurricanes directory.
    data.variable_channels = 112
    data.output_channels = 2

    # model
    model = config.model
    model.name = "cncsnpp"
    model.scale_by_sigma = False
    model.ema_rate = 0.9999
    model.normalization = "GroupNorm"
    model.nonlinearity = "swish"
    model.nf = 128
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 4
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = True
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = "biggan"
    model.progressive = "none"
    model.progressive_input = "residual"
    model.progressive_combine = "sum"
    model.attention_type = "ddpm"
    model.embedding_type = "positional"
    model.init_scale = 0.0
    model.fourier_scale = 16
    model.conv_size = 3
    model.loc_spec_channels = 0

    # sampling
    sampling = config.sampling
    sampling.method = "pc"
    sampling.predictor = "euler_maruyama"
    sampling.corrector = "none"

    return config
