import cv2


def script_mark(image, script):
    x1 = script.x1
    y1 = script.y1
    x2 = script.x2
    y2 = script.y2

    cv2.putText(image, script.code, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0))

    return image
