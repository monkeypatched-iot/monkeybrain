import numpy as np

# Function to calculate distance from point to a line segment
def point_to_segment_distance(point, segment_start, segment_end):
    px, py = point[:2]
    x1, y1 = segment_start
    x2, y2 = segment_end

    # Convert px and py to floats if they are strings
    px = float(px)
    py = float(py)

    # Vector calculations
    seg_vec = np.array([x2 - x1, y2 - y1])
    pt_vec = np.array([px - x1, py - y1])

    seg_len_sq = np.dot(seg_vec, seg_vec)

    # Project the point onto the segment
    if seg_len_sq == 0:
        # segment_start == segment_end (degenerate case, distance to point)
        return np.linalg.norm(pt_vec)

    projection = np.dot(pt_vec, seg_vec) / seg_len_sq
    projection = np.clip(projection, 0, 1)  # restrict to [0, 1]

    # Find the closest point on the segment
    closest_point = np.array([x1, y1]) + projection * seg_vec
    closest_dist = np.linalg.norm(np.array([px, py]) - closest_point)

    return closest_dist

def convert_to_segments(points):
    segments = []
    for i in range(len(points) - 1):
        # Create a tuple of consecutive points
        segment = (points[i], points[i+1])
        segments.append(segment)
    return segments

# Find keyframe points within 1 foot of any path
def find_keyframes_along_path(keyframe_points, path_segments, max_distance=3.0):
    nearby_points = []

    # Iterate over each point
    for point in keyframe_points:
        # Iterate over each segment
        for seg_start, seg_end in path_segments:
            
            dist = point_to_segment_distance(point, seg_start, seg_end)
 
            if dist <= max_distance:
                nearby_points.append(point)
                # No need to check other segments if this point is close enough
                break

    return nearby_points