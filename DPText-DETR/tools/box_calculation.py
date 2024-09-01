import json
from shapely.geometry import Polygon
from shapely.validation import explain_validity

def load_json(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Failed to load or parse the JSON file: {e}")
        return None

def validate_and_correct_polygon(polygon):
    if not polygon.is_valid:
        print(f"Polygon is invalid: {explain_validity(polygon)}")
        corrected_polygon = polygon.buffer(0)
        if corrected_polygon.is_valid:
            print("Polygon has been corrected.")
            return corrected_polygon
        else:
            print("Failed to correct the polygon.")
            return None
    return polygon

def calculate_iou(poly1, poly2):
    if poly1 is None or poly2 is None:
        return 0
    if not poly1.intersects(poly2):
        return 0
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    if union == 0:
        return 0
    return intersection / union

def count_overlapping_boxes(file1_data, file2_data, iou_threshold=0.8):
    overlap_count = 0
    file1_polygons = {}
    file2_polygons = {}

    for item in file1_data:
        image_name = item['image_name']
        if image_name not in file1_polygons:
            file1_polygons[image_name] = []
        polygon = validate_and_correct_polygon(Polygon(item['polys']))
        if polygon:
            file1_polygons[image_name].append(polygon)

    for item in file2_data:
        image_name = item['image_name']
        if image_name not in file2_polygons:
            file2_polygons[image_name] = []
        polygon = validate_and_correct_polygon(Polygon(item['polys']))
        if polygon:
            file2_polygons[image_name].append(polygon)

    # Compare polygons only within the same image_name
    for image_name in file1_polygons:
        if image_name in file2_polygons:
            for poly1 in file1_polygons[image_name]:
                for poly2 in file2_polygons[image_name]:
                    if calculate_iou(poly1, poly2) >= iou_threshold:
                        overlap_count += 1
    return overlap_count

def main():
    json_path_1 = '/home/ysjeong/workspace/OCR/DPText-DETR/output/r_50_poly/totaltext/finetune/inference/text_results.json'
    json_path_2 = '/home/ysjeong/workspace/OCR/Bridging_Spotting/Bridging-Text-Spotting/output/Bridge/TotalText/R_50_Polygon/inference/text_results.json'
    data1 = load_json(json_path_1)
    data2 = load_json(json_path_2)

    if data1 is None or data2 is None:
        print("One or both of the data files could not be loaded.")
        return

    count_file1 = len(data1)
    count_file2 = len(data2)
    overlap_count = count_overlapping_boxes(data1, data2)

    print(f"Number of boxes in file 1: {count_file1}")
    print(f"Number of boxes in file 2: {count_file2}")
    print(f"Number of overlapping boxes: {overlap_count}")

if __name__ == "__main__":
    main()
