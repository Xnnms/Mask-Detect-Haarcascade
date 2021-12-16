import cv2

face_mask = cv2.CascadeClassifier("cascadedownload/cascade.xml")
faceCascade = cv2.CascadeClassifier("haarcascade/haarcascade_frontalface_default.xml")
noseCascade = cv2.CascadeClassifier("haarcascade/Nariz.xml")

cap = cv2.VideoCapture(0)
mask_on = False


 
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
     
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    wajah = faceCascade.detectMultiScale(gray, 1.1, 5)

    for(x, y, w, h) in wajah:
    	roi_gray = gray[y:y+h, x:x+w]
    	roi_color = frame[y:y+h, x:x+w]
    	if mask_on:(
    		cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 1),
    		cv2.putText(frame,'Mask On',(x,y),1,2,(0,255,0),2),
            
        )
    	else:(
    		cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255),1),
            cv2.putText(frame,'Mask Off',(x,y),1,2,(0,0,255),2),
        
        )

    # gambar kotak di hidung
    hidung = noseCascade.detectMultiScale(gray, 1.18, 35,)
    for(sx, sy, sw, sh) in hidung:
    	cv2.rectangle(frame, (sx,sy), (sx+sw, sy+sh), (255,0,0),2),
    	cv2.putText(frame,'Hidung',(sx,sy),1,1,(255,0,0),1)
        
        
    face = face_mask.detectMultiScale(gray,1.1,4)
    for(x,y,w,h) in face:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color=frame[y:y+h, x:x+w]
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1),
        cv2.putText(frame,'masker',(x,y),1,1,(255,0,0),1),
        


    
    if(len(face) > 0 and len(hidung) == 0 ):
        mask_on = False  
    elif(len(hidung) == 0):
        mask_on = True 
    else:
        mask_on = False
        
         


    # Display the resulting frame

    cv2.imshow('video',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()