import argparse
import time
from multiprocessing import Event
from queue import Queue
from threading import Thread

import cv2
from ultralytics import YOLO


def parse():
    parser = argparse.ArgumentParser(description="Запуск yolov8s-pose на нескольких потоках")
    parser.add_argument("-t", "--num_threads", type=int, default=1,
                        help="Количество потоков для многопоточного режима.")
    parser.add_argument("video_path", type=str, help="Путь к входному видеофайлу.")
    parser.add_argument("output_file", type=str, help="Имя выходного видеофайла.")
    args = parser.parse_args()
    return args.num_threads, args.video_path, args.output_file


def predict_work(in_queue, out_queue, stop_event):
    model = YOLO("yolov8s-pose.pt")
    while not stop_event.is_set():
        frame = in_queue.get()
        if frame is None:  
            break
        out_queue.put(model.predict(frame, verbose=False, device="cpu")[0].plot())


if __name__ == "__main__":
    num_threads, video_path, output_file = parse()

    start_time = time.time()

    in_video = cv2.VideoCapture(video_path)

    stop_event = Event()
    in_queues = [Queue() for _ in range(num_threads)]
    out_queues = [Queue() for _ in range(num_threads)]
    threads = [Thread(target=predict_work, args=(in_queues[i], out_queues[i], stop_event)) for i in range(num_threads)]
    for thread in threads:
        thread.start()

    num_frames = 0
    while True:
        success, frame = in_video.read()
        if not success:
            break
        in_queues[num_frames % num_threads].put(frame)
        num_frames += 1

    for in_queue in in_queues:
        in_queue.put(None)

    out_video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 30,
                                (int(in_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                 int(in_video.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    for i in range(num_frames):
        frame = out_queues[i % num_threads].get()
        out_video.write(frame)

    stop_event.set()
    for thread in threads:
        thread.join()
    in_video.release()
    out_video.release()

    end_time = time.time()

    print(f"Elapsed time: {round(end_time - start_time, 2)}")
