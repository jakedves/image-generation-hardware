import cv2
import os


def extract_frames(folder, filename):
    output_directory = folder + filename[:-4] + '/'

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    capture = cv2.VideoCapture(folder + filename)
    if not capture.isOpened():
        print(f"Cannot open video file: {filename}")
        return

    index = 0
    frame_read = True
    while frame_read:
        frame_read, frame = capture.read()

        if not frame_read:
            break

        output_filename = output_directory + f"{index}.png"
        cv2.imwrite(output_filename, frame)

        index += 1
        if index % 20 == 0:
            print(f"Saved {index} frames")

    capture.release()


if __name__ == "__main__":
    directory = 'datasets/frame-interpolation/sjtu-4k/'

    for file in os.listdir(directory):
        if file.endswith('.mp4'):
            print(file)
            extract_frames(directory, file)

