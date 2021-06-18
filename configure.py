if 0:
    STOI = {
        '<sos>': 190,
        '<eos>': 191,
        '<pad>': 192,
    }


    image_size = 224
    vocab_size = 193
    max_length = 300 #275

else:
    STOI = {
        '<sos>': 36,
        '<eos>': 37,
        '<pad>': 38,
    }


image_size_ = 320
vocab_size = 39
max_length = 400 #391x

patch_size = 16
pixel_pad = 3
pixel_stride = 4
num_pixel = (patch_size // pixel_stride) ** 2
pixel_scale = 0.8

max_patch_row_col = 500
max_num_patch = 600

    