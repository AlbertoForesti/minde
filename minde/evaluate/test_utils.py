import mutinfo.distributions.images.geometric as geometric

def get_image_generation_fn(drawing_fn_name, coord_fn_name, image_shape, min_size, max_size=None):
    
    drawing_fn = getattr(geometric, drawing_fn_name)
    coord_fn = getattr(geometric, coord_fn_name)
    def generate_image(x, y):
        return (
        drawing_fn(coord_fn(x, min_size=(0.2, 0.2)), image_shape),
        drawing_fn(coord_fn(y, min_size=(0.2, 0.2)), image_shape)
        )
    return generate_image