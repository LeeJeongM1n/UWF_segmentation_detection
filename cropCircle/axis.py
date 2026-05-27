import pandas as pd
import numpy as np
import cv2
import os

# 1. Load the original dataset
df = pd.read_csv('/mnt/richul_FM/UWF_seg_det/datasets/YOLO/det_output/train_resnet50/sigma20/inference/tta_circle_mean/internal_testDataset.csv')

# 2. Setup output directory
output_dir = '/mnt/richul_FM/UWF_seg_det/datasets/YOLO/det_output/train_resnet50/sigma20/inference/tta_circle_mean//test_axis/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def parse_coords(s):
    return tuple(map(int, s.split(',')))

torsion_degrees = []
torsion_types = []

for index, row in df.iterrows():
    # Parse coordinates
    m_center = parse_coords(row['macula_center'])
    d_center = parse_coords(row['disc_center'])
    xm, ym = m_center
    xd, yd = d_center
    
    # Calculate Angle (y increases downwards)
    dx = abs(xm - xd)
    dy = ym - yd
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)
    
    # Determine type
    t_type = "extratorsion" if angle_deg > 0 else "intratorsion"
    torsion_degrees.append(round(angle_deg, 2))
    torsion_types.append(t_type)
    
    # 3. Image Processing
    img_path = os.path.join("./test/", row['image_fname'])
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        
        # Draw Short Horizontal Line (Bright Yellow)
        # From disc center to the x-alignment of the macula
        cv2.line(img, d_center, (xm, yd), (0, 255, 255), 2)
        
        # Draw Axis Line to Macula (Magenta)
        cv2.line(img, d_center, m_center, (255, 0, 255), 2)
        
        # Draw points
        cv2.circle(img, m_center, 4, (255, 0, 0), -1) # Macula: Blue
        cv2.circle(img, d_center, 4, (0, 255, 0), -1) # Disc: Green
        
        # Add text annotation
        text = f"{angle_deg:.2f} deg ({t_type})"
        cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Save output image
        cv2.imwrite(os.path.join(output_dir, f"axis_{row['image_fname']}"), img)

# 4. Save updated CSV
df['torsion_degree'] = torsion_degrees
df['torsion_type'] = torsion_types
df.to_csv(os.path.join(output_dir, 'csv_axis.csv'), index=False)

print("Processing complete. Files saved in './test_axis/'")