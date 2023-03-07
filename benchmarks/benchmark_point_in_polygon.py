import time
import numpy as np

def point_in_polygon_matplotlib(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    pass

def point_in_polygon_cubao(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    '''
    it's migrated from matplotlib
    '''
    from fast_crossing import point_in_polygon
    pass

def point_in_polygon_polygons(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    from fast_crossing import point_in_polygon

def point_in_polygon_shapely(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    from fast_crossing import point_in_polygon

def load_points(path: str):
    pass

def load_polygon(path: str):
    if path.endswith(('.npy', '.pcd')):
        return load_points(path)
    pass

def write_mask(mask: np.ndarray, path: str):
    pass

def wrapping(fn):
    def wrapped_fn(input_points: str, input_polygon: str, output_path: str):
        points = load_points(input_points)
        polygon = load_polygon(input_polygon)
        mask = fn(points, polygon)
        write_mask(mask, output_path)
    return wrapped_fn

if __name__ == '__main__':
    import fire
    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire({
        'matplotlib': wrapping(point_in_polygon_matplotlib),
        'cubao': wrapping(point_in_polygon_cubao),
        'polygons': wrapping(point_in_polygon_polygons),
        'shapely': wrapping(point_in_polygon_shapely),
    })