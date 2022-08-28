import cv2


def face_capture():
    cascade_path = '/home/oleg/Python/Practice/Python_Today/Face_Recognition_280822/filters/' \
                   'haarcascade_frontalface_default.xml'

    clf = cv2.CascadeClassifier(cascade_path)
    camera = cv2.VideoCapture('man.mp4')

    while True:
        _, frame = camera.read()

        image = cv2.imread('man.mp4')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = clf.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, width, height) in faces:
            cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 255, 0), 2)

        cv2.imshow("Faces", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


def main():
    face_capture()


if __name__ == '__main__':
    main()
