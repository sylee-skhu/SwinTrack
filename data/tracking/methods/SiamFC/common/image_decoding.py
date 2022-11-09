import torchvision.io


class SiamFCImageDecodingProcessor:
    def __init__(self, post_processor=None):
        self.post_processor = post_processor

    def __call__(self, z_image_path, z_bbox, x_image_path, x_bbox, is_positive):
        z_image = torchvision.io.read_image(z_image_path)
        if z_image.shape[0] == 4:
            z_image = z_image[:3]
        elif z_image.shape[0] == 1:
            z_image = z_image.repeat(3, 1, 1)

        if z_image_path != x_image_path:
            x_image = torchvision.io.read_image(x_image_path)
            if x_image.shape[0] == 4:
                x_image = x_image[:3]
            elif x_image.shape[0] == 1:
                x_image = x_image.repeat(3, 1, 1)
        else:
            x_image = z_image
        data = (z_image, z_bbox, x_image, x_bbox, is_positive)
        if self.post_processor is not None:
            return self.post_processor(*data)
        else:
            return data

