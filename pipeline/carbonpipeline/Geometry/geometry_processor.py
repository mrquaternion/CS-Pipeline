from .geometry import Geometry, GeometryType

class GeometryProcessor: 
    @staticmethod
    def process_geometry(data: list) -> tuple[Geometry, any]:
        """
        Examples:
            Point = [4.0, 3.6, 12.1, ...]
            Polygon = [
                            [4.0, 3.6, 12.1, ...],
                            [1.9, 2.0, 2.4, ...],
                            ...
                        ]
            MultiPolygon = [
                                [
                                    [4.0, 3.6, 12.1, ...],
                                    [1.9, 2.0, 2.4, ...],
                                    ...
                                ],
                                [
                                    ...
                                ],
                                ...
                            ]
        """
        geometry = Geometry(data=data)
        print(geometry.type_signature)
        geometry.validate_coordinates(data=data)
        match geometry.geom_type:
            case GeometryType.POINT:
                region = GeometryProcessor._get_point_outer_bounds(geometry.data)
                return (geometry, region)
            case GeometryType.POLYGON:
                rect_region = GeometryProcessor._get_rect_region_covering_polygon(geometry.data)
                return (geometry, rect_region)
            case GeometryType.MULTIPOLYGON:
                rect_regions = GeometryProcessor._get_polys_regions(geometry.data)
                return (geometry, rect_regions)
            case GeometryType.UNKNOWN:
                raise TypeError("Unsupported geometry depth.")
        
    @staticmethod
    def _get_point_outer_bounds(point: list[float]) -> list[float]:
        offset = 0.125
        lat, lon = point
        return [lat + offset, lon - offset, lat - offset, lon + offset]

    @staticmethod
    def _get_rect_region_covering_polygon(poly: list[list[float]]) -> list[float]:
        """
        ERA5 asks for 4 points:
            N = max latitude
            W = min longitude
            S= min latitude
            E = max longitude
        """
        lons = [coord[0] for coord in poly]
        lats = [coord[1] for coord in poly]
        return [max(lats), min(lons), min(lats), max(lons)]

    @staticmethod
    def _get_polys_regions(polys: list[list[list[float]]]) -> tuple[dict[int, list[list[float]]], dict[int, list[float]]]:
        """
            enum_polys = {
                            1: [
                                [lat, lon], 
                                [lat, lon], 
                                ...
                            ]
                        }
            rect_regions = {
                            1: [N, W, S, E], 
                            2: [N, W, S, E], 
                            ...
                        }
        """
        rect_regions = {}
        for i, poly in enumerate(polys):
            # We find the 4 points that makes a rect covering the polygon
            rect_regions[i] = GeometryProcessor._get_rect_region_covering_polygon(poly)
        return rect_regions
    