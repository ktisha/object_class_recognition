import cv2


def main():
    img = cv2.imread('../data/train/cat.0.jpg')
    cv2.imshow('cat', img)
    cv2.waitKey()


if __name__ == '__main__':
    main()