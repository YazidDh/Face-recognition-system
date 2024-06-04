import face_recognition
import os ,sys
import cv2
import numpy as np
import math
import predictMask
import predict_antispoofing
import predict_age_gen

def face_confidence(face_distance, face_math_threshhold=0.6):
    range = (1.0 - face_math_threshhold)
    linear_val = (1.0 - face_distance) / (range * 2.0)
    if face_distance > face_math_threshhold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'

class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()
    def encode_faces(self):
        for image in os.listdir("AdiminPlat/fr_admin_plat/images"):
            face_image = face_recognition.load_image_file(f'AdiminPlat/fr_admin_plat/images/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
        print(self.known_face_names)

    def run_recognition(self):
        video_cap = cv2.VideoCapture(0)
        #video_cap = cv2.VideoCapture("http://192.168.137.173:5000/")

        if not video_cap.isOpened():
            sys.exit("Video source not found..")
        while True:
            ret, frame = video_cap.read()
            res_spoofing = predict_antispoofing.predict_antispoofing(frame)

            # print(res_spoofing[0])
            if (res_spoofing[0] == 0):
                # spoof
                cv2.putText(frame, "Spoofing", (10, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 1)

            else :
                if self.process_current_frame:
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = small_frame[:, :, ::-1]
                    self.face_locations = face_recognition.face_locations(rgb_small_frame)
                    self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)
                    self.face_names = []
                    for face_encoding in self.face_encodings:
                        matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                        name = 'Unknown'
                        confidence = 'Unknown'
                        face_dist = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_dist)
                        if matches[best_match_index]:
                            name = self.known_face_names[best_match_index]
                            confidence = face_confidence(face_dist[best_match_index])

                        self.face_names.append(f'{name}')

                self.process_current_frame = not self.process_current_frame
                for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                    input_dataset = frame[top+5:bottom+5, left+5:right+5]
                    res_mask = predictMask.predictMask(input_dataset)
                    dernier_point = name.rfind('.')
                    name = name[:dernier_point]
                    age, gender = predict_age_gen.predict(input_dataset)
                    cv2.putText(frame, gender + "," + str(age), (left + 6, bottom - 195), cv2.FONT_HERSHEY_DUPLEX, 0.8,
                                (0, 0, 255), 1)
                    #print(age,gender)
                    if (res_mask[0] == 1):
                        # with mask
                        cv2.putText(frame, name + "with mask", (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8,(255, 255, 255), 1)
                    else :
                        cv2.putText(frame, name , (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) == ord('q'):
                break
        video_cap.release()
        cv2.destroyAllWindows()


fr = FaceRecognition()
fr.run_recognition()
