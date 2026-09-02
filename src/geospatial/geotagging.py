import math
import logging
from typing import List, Tuple
from src.utils.detection import Detection, CoordinateType
from src.geospatial.metadata import SonarMetadata
from src.utils.config import Config

class GeospatialEngine:
    def __init__(self, config: Config, image_shape: Tuple[int, int]):
        self.config = config.geospatial
        self.image_shape = image_shape # (H, W)

    def process(self, detections: List[Detection], metadata: SonarMetadata = None) -> List[Detection]:
        for det in detections:
            # First, compute physical dimensions if resolution is known
            if metadata and metadata.across_track_resolution and metadata.along_track_resolution:
                w_px = det.bbox[2] - det.bbox[0]
                h_px = det.bbox[3] - det.bbox[1]
                det.width_m = w_px * metadata.across_track_resolution
                det.height_m = h_px * metadata.along_track_resolution

            if metadata:
                det.timestamp = metadata.timestamp
                det.ping_id = metadata.ping_id

                # Check if we have required fields for accurate GPS
                required_fields = self.config.coordinate_type_required_fields
                has_all_required = all(getattr(metadata, field, None) is not None for field in required_fields)
                
                has_lat_lon = metadata.latitude is not None and metadata.longitude is not None

                if has_all_required:
                    # Fake affine transform for MVP (flat-Earth ENU approximation)
                    # Real application would use proper coordinate reference systems (pyproj)
                    lat, lon = self._compute_flat_earth_pos(
                        metadata.latitude, 
                        metadata.longitude, 
                        metadata.heading, 
                        det.bbox, 
                        metadata.across_track_resolution
                    )
                    det.latitude = lat
                    det.longitude = lon
                    det.coordinate_type = CoordinateType.GPS_ACCURATE
                    
                elif has_lat_lon:
                    # We have lat/lon of the vessel, but lack heading or resolution to project it
                    # Just use vessel lat/lon as estimated position of the target
                    det.latitude = metadata.latitude
                    det.longitude = metadata.longitude
                    det.coordinate_type = CoordinateType.GPS_ESTIMATED
                    
                else:
                    det.coordinate_type = CoordinateType.IMAGE_SPACE
            else:
                det.coordinate_type = CoordinateType.IMAGE_SPACE
                
        return detections

    def _compute_flat_earth_pos(self, vessel_lat: float, vessel_lon: float, heading_deg: float, bbox: List[float], res: float) -> Tuple[float, float]:
        # Center of bbox
        x_center = (bbox[0] + bbox[2]) / 2.0
        
        # Distance from nadir (assume nadir is center of image width for simple side-scan)
        # If nadir is center, and swath is W:
        nadir_x = self.image_shape[1] / 2.0
        dist_px = x_center - nadir_x
        dist_m = dist_px * res
        
        # Simple projection: 1 degree lat approx 111,111 meters
        # This is a very rough approximation strictly for the MVP logic
        heading_rad = math.radians(heading_deg)
        
        # If target is to the right (positive dist_m), bearing is heading + 90
        bearing_rad = heading_rad + math.radians(90 if dist_m > 0 else -90)
        dist_m = abs(dist_m)
        
        delta_lat = math.cos(bearing_rad) * dist_m / 111111.0
        delta_lon = math.sin(bearing_rad) * dist_m / (111111.0 * math.cos(math.radians(vessel_lat)))
        
        return vessel_lat + delta_lat, vessel_lon + delta_lon
