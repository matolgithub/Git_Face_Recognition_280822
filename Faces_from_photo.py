import cv2


def viewImage(image, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def face_recognaze():
    image_path = "girls.webp"

    cascade_path = '/home/oleg/Python/Practice/Python_Today/Face_Recognition_280822/filters/haarcascade_frontalface_default.xml'

    face_cascade = cv2.CascadeClassifier(cascade_path)
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(10, 10)
    )
    faces_detected = "Лиц обнаружено: " + format(len(faces))
    print(faces_detected)
    # Рисуем квадраты вокруг лиц
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
    viewImage(image, faces_detected)


if __name__ == '__main__':
    face_recognaze()