
import face_recognition
import cv2
import numpy as np
from faces import known_cface_encodings, known_cface_names, known_mface_encodings, known_mface_names

camera = cv2.VideoCapture(0)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Resize frame of video to 1/2 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
                        
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                cmatches = face_recognition.compare_faces(known_cface_encodings, face_encoding)
                mmatches = face_recognition.compare_faces(known_mface_encodings, face_encoding)
                name = "Unknown"
                # Or instead, use the known face with the smallest distance to the new face
                cface_distances = face_recognition.face_distance(known_cface_encodings, face_encoding)
                cbest_match_index = np.argmin(cface_distances)
                mface_distances = face_recognition.face_distance(known_mface_encodings, face_encoding)
                mbest_match_index = np.argmin(mface_distances)
                if cmatches[cbest_match_index] and not mmatches[mbest_match_index]:
                    name = known_cface_names[cbest_match_index] +" - has Criminal Record!!"
                if mmatches[mbest_match_index] and not cmatches[cbest_match_index]:
                    name = known_mface_names[mbest_match_index] + "- Missing Person!"
                if mmatches[mbest_match_index] and cmatches[cbest_match_index]:
                    name = known_mface_names[mbest_match_index] + "- Missing and has Criminal Record too!"
                face_names.append(name)
                                       
                # Display the results
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                    top *= 2
                    right *= 2
                    bottom *= 2
                    left *= 2                             
                               
                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 245, 0), 2)

                    # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 245, 0), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_TRIPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
                          
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                            
                yield (b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
cv2.waitKey(0)
cv2.destroyAllWindows()