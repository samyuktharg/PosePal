import cv2
import mediapipe as mp
import time
import os
import numpy as np

# --- AESTHETICS ---
COLOR_BG = (255, 230, 230)      # Light Lavender
COLOR_ACCENT = (180, 130, 255)  # Hot Pink
COLOR_TIMER = (255, 255, 255)   # White for the timer circle

# --- SETUP ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, max_num_hands=1)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# --- VARIABLES ---
trigger_start_time = 0
photo_taken_time = 0
last_photo = None 

# Animation Settings
ANIMATION_DURATION = 1.0 
THUMB_WIDTH = 200 
THUMB_HEIGHT = 112 

print("Cute Booth (Peace Sign Only) Started...")

while True:
    success, img = cap.read()
    if not success: break
    img = cv2.flip(img, 1)
    
    # 1. Create a "Clean Copy" for saving (No UI will be drawn on this)
    clean_frame = img.copy()
    
    h_screen, w_screen, _ = img.shape

    # 2. Hand Tracking
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    pose_detected = False
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w_screen), int(lm.y * h_screen)
                lm_list.append([id, cx, cy])
            
            if len(lm_list) != 0:
                fingers = []
                tips = [8, 12, 16, 20]
                knuckles = [6, 10, 14, 18]
                
                # Check Fingers (Index, Middle, Ring, Pinky)
                # If Tip is higher (smaller y) than Knuckle -> 1 (UP)
                for i in range(4):
                    if lm_list[tips[i]][2] < lm_list[knuckles[i]][2]: 
                        fingers.append(1)
                    else: 
                        fingers.append(0)
                
                # PEACE SIGN LOGIC
                # Index (fingers[0]) & Middle (fingers[1]) must be UP (1)
                # Ring (fingers[2]) & Pinky (fingers[3]) must be DOWN (0)
                if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0:
                    pose_detected = True

    # 3. Trigger & Animation Logic
    current_time = time.time()
    
    # --- ANIMATION MODE (Waiting between photos) ---
    if last_photo is not None and (current_time - photo_taken_time) < ANIMATION_DURATION:
        t = (current_time - photo_taken_time) / ANIMATION_DURATION
        
        # Shrink and Move Animation
        curr_w = int(w_screen - (w_screen - THUMB_WIDTH) * t)
        curr_h = int(h_screen - (h_screen - THUMB_HEIGHT) * t)
        target_x, target_y = 50, h_screen - THUMB_HEIGHT - 50
        
        curr_x = int(0 + (target_x - 0) * t)
        curr_y = int(0 + (target_y - 0) * t)
        
        small_photo = cv2.resize(last_photo, (curr_w, curr_h))
        
        # Overlay the flying photo onto the screen
        if curr_y + curr_h < h_screen and curr_x + curr_w < w_screen:
            img[curr_y:curr_y+curr_h, curr_x:curr_x+curr_w] = small_photo
            cv2.rectangle(img, (curr_x, curr_y), (curr_x+curr_w, curr_y+curr_h), (255,255,255), 5)
            
    else:
        if last_photo is not None: last_photo = None 

        # --- DETECTION MODE ---
        if pose_detected:
            if trigger_start_time == 0:
                trigger_start_time = current_time
            
            elapsed = current_time - trigger_start_time
            
            # Draw Timer on 'img' (User sees this)
            center_x, center_y = 640, 360
            radius = 100
            # Draw white ring
            cv2.circle(img, (center_x, center_y), radius, COLOR_TIMER, 5)
            # Draw filling arc
            angle = int((elapsed / 2.0) * 360)
            cv2.ellipse(img, (center_x, center_y), (radius, radius), -90, 0, angle, COLOR_ACCENT, -1)
            
            if elapsed > 2.0:
                # --- TAKE PHOTO NOW ---
                
                # 1. Add the Cute Border to the CLEAN frame
                cv2.rectangle(clean_frame, (0, 0), (1280, 720), COLOR_BG, 30)
                
                # 2. Save the CLEAN frame (No timer!)
                last_photo = clean_frame.copy() 
                filename = f"cute_snap_{int(time.time())}.jpg"
                cv2.imwrite(filename, last_photo)
                print(f"Saved {filename}")
                
                # 3. Start Animation
                photo_taken_time = current_time
                trigger_start_time = 0
                
                # 4. Flash Effect
                cv2.rectangle(img, (0,0), (w_screen, h_screen), (255, 255, 255), cv2.FILLED)
                
        else:
            trigger_start_time = 0

    # 4. Final Display
    # Draw border on screen so user knows what the photo will look like
    cv2.rectangle(img, (0, 0), (1280, 720), COLOR_BG, 30)
    
    cv2.imshow("Cute Booth (Peace)", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()