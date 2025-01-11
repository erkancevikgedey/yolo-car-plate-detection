import cv2
from ultralytics import YOLO
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import easyocr

def make_square(im, size=1280, fill_color=(0, 0, 0)):
    h, w = im.shape[:2]
    
    new_im = np.zeros((size, size, 3), dtype=np.uint8)
    new_im[:, :] = fill_color
    
    aspect_ratio = w / h
    if aspect_ratio > 1:
        new_w = size
        new_h = int(size / aspect_ratio)
    else:
        new_h = size
        new_w = int(size * aspect_ratio)
    
    resized_im = cv2.resize(im, (new_w, new_h))
    
    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2
    new_im[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_im
    
    return new_im


plaka_model = YOLO('best_c_ultra_yenidt.pt')
plaka_harf_model = YOLO('bestu_c_harf_siyah.pt')
reader = easyocr.Reader(['en'])

plaka_harf_model.info()

vidcap = cv2.VideoCapture('yol.mp4')


success,image = vidcap.read()
count = 0

while success:
    success,image = vidcap.read()
    image_duzen_dongu = make_square(image)
    results = plaka_model(image_duzen_dongu, imgsz=1280, conf=0.8)
    for i, result in enumerate(results):
        plakalar = []
        for detection in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            plakalar.append([x1, y1, x2, y2, score])
        print(plakalar)
        cv2.imshow("Cameras",image_duzen_dongu)
        cv2.waitKey(1) 
        for plaka in plakalar:
            x1, y1, x2, y2, score = plaka
            im_bgr = result.plot()  
            im_rgb = Image.fromarray(im_bgr[..., ::-1]) 
            cls = int(result.boxes[i].cls)
            label = f'Plaka {score}'
           
            cropped_img = image_duzen_dongu[int(y1):int(y2), int(x1):int(x2)]
            cv2.rectangle(image_duzen_dongu, (int(x2), int(y2)), (int(x1), int(y1)), (255, 255, 0), 2)
            cv2.putText(image_duzen_dongu, label, (int(x2), int(y2) - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 0), 1)                
           
            cv2.imshow("Cameras",image_duzen_dongu)

            

            harf_sonuclar = plaka_harf_model(cropped_img)
            tr_sil = cropped_img[:,7:] 
            license_plate_crop_gray = cv2.cvtColor(tr_sil, cv2.COLOR_BGR2GRAY) 
            license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)[1]
            

            cv2.imshow("Cameras2",license_plate_crop_gray)

            ocr_sonuc = reader.readtext(license_plate_crop_thresh)
            print(f"OCR ÇIKTISI: {ocr_sonuc}")
            if len(ocr_sonuc) > 0: 
                cv2.putText(image_duzen_dongu, f"OCR: {ocr_sonuc[0][1]}", (int(x2), int(y2) - 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 0), 1)
                cv2.imshow("Cameras",image_duzen_dongu)

            for ih, harf_sonuc in enumerate(harf_sonuclar):
                boxes = np.array(harf_sonuc.boxes.data.tolist())
                if(boxes.size<=0):
                    print("Plaka okunamadı")
                    break
                
                sorted_indices = np.argsort(boxes[:, 0])
                sorted_boxes = boxes[sorted_indices]
                plaka = ""
                for box in sorted_boxes:
                    harf_x1, harf_y1, harf_x2, harf_y2, harf_score, harf_class_id = box
                    print(f"Harf: {harf_sonuc.names[int(harf_class_id)]}, x1: {harf_x1}, y1: {harf_y1}, x2: {harf_x2}, y2: {harf_y2}, Skor: {harf_score}")
                    plaka += harf_sonuc.names[int(harf_class_id)]

                cv2.putText(image_duzen_dongu, f"Plaka Tam: {plaka}", (int(x2), int(y2) + 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("Cameras",image_duzen_dongu)
                

                print(f"Plaka Tam: {plaka}")
            cv2.waitKey(1) 

    count += 1

