from enum import Enum
from typing import Union

class GeometryType(Enum):
    POINT = "Point"
    POLYGON = "Polygon"
    MULTIPOLYGON = "MultiPolygon"
    UNKNOWN = "Unknown"

class Geometry:
    def __init__(self, data):
        if not isinstance(data, list):
            raise TypeError("Expected a list, but received a different type. Verify your JSON file.")
        
        self.depth = self._get_depth(data)
        self.data = self._flatten(data) # Let's flatten the array beforehand
        self.type_signature = self._get_type_signature(data)
        self.geom_type = self._infer_geom_type()

    def validate_coordinates(self, data):
        if isinstance(data, list) and all(isinstance(item, (int, float)) for item in data):
            if len(data) != 2:
                raise ValueError(f"Invalid coordinate pair. Expected 2 elements but received {len(data)}")
            return
        elif isinstance(data, list):
            for item in data:
                self.validate_coordinates(item)
        else:
            raise TypeError(f"Invalid element in coordinate structure: {self.data}")
    
    def _flatten(self, lst) -> list[list[list[Union[float, int]]]]:
        depth = self.depth
        result = lst
        while depth > 3:
            result = [item for subresult in result for item in subresult]
            depth -= 1
        self.depth = depth
        return result

    def _get_depth(self, lst):
        if isinstance(lst, list) and lst:
            return 1 + max(self._get_depth(item) for item in lst)
        if isinstance(lst, list):
            return 1
        else:
            return 0

    def _get_type_signature(self, lst):
        if isinstance(lst, list):
            if lst:
                return "[" + self._get_type_signature(lst[0]) + "]"
            else:
                return "[?]"
        else:
            return type(lst).__name__

    def _infer_geom_type(self):
        if self.depth == 1:
            return GeometryType.POINT
        elif self.depth == 2:
            return GeometryType.POLYGON
        elif self.depth == 3:
            return GeometryType.MULTIPOLYGON
        else:
            return "Unknown"
        
    def __repr__(self):
        return f"Geometry={self.geom_type}, depth={self.depth}, signature={self.type_signature}"