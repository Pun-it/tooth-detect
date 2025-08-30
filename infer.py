import cv2
from ultralytics import YOLO

model_path = r"tooth-detect\best.pt"   
image_path = r"tooth-detect/test.jpg"                
save_path  = r"tooth-detect/output.jpg"  

model = YOLO(model_path)

results = model(image_path, conf=0.35)  

annotated_img = results[0].plot()


# cv2.imshow("YOLO Detection", annotated_img)
# cv2.waitKey(0)   
# cv2.destroyAllWindows()

# Save result
cv2.imwrite(save_path, annotated_img)
print(f"Saved to : {save_path}")
