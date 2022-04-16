import cv2

def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, clf):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray, scaleFactor, minNeighbors)
    id = 0
    confidence = 100
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        id, confidence = clf.predict(gray[y: y + h, x: x + w])

        if confidence > 50:
            id = 0
        
    return img, id, round(100 - confidence)

def conclude(name_results):
    names_appeared = {name_result[0]: 0 for name_result in name_results}
    if len(names_appeared) == 1:
        return list(names_appeared)[0]
    else:
        for name_result in name_results:
            names_appeared[name_result[0]] += 1
        frequencies = list(names_appeared.values())
        most_appeared = list(names_appeared.keys())[frequencies.index(max(frequencies))]
        return most_appeared

faceCascade = cv2.CascadeClassifier("cascade/haarcascade_frontalface_default.xml")
users = ['Unknown', 'Cheep', 'Champ', 'Peak', 'Tonnam']
name_results = []

username = input('Please input your username: ')
if not username in users or username == 'Unknown':
    print('Invalid username! Try again.')
else:
    cap = cv2.VideoCapture(0)
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read('train/classifier.xml')

    for takes in range(60):
        ret, frame = cap.read()
        frame, id, confidence = draw_boundary(frame, faceCascade, 1.1, 10, (0, 255, 0), clf)
        name_results.append([users[id], confidence])
        cv2.imshow('frame', frame)
        takes += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

    if conclude(name_results) == username:
        print('Login successful')
    else:
        print('Authentication failed. Try again.')
        
    cv2.destroyAllWindows()
