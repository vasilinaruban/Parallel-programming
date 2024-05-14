import queue
import time
import threading
import cv2
import logging
import argparse

class Sensor:
    def get(self):
        raise NotImplementedError('Subclasses must implement method get()')

class SensorCam(Sensor):
    def __init__(self, name, resol):
        self._cap = cv2.VideoCapture(name)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, resol[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resol[1])
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def get(self):
        ret, frame = self._cap.read()
        if not ret:
            logging.error(Exception('Unable to read the input.'))
            raise Exception("Failed to capture frame from camera")
        return frame

    def __del__(self):
        self._cap.release()

class SensorX(Sensor):
    def __init__(self, delay):
        self._delay = delay
        self._data = 0

    def get(self):
        time.sleep(self._delay)
        self._data += 1
        return self._data

def sensor_loop(sensor, q):
    while True:
        if q.full():
            q.get_nowait()
        q.put_nowait(sensor.get())

def main():

    logger = logging.basicConfig(filename = "./log/errors.log", level = logging.ERROR, format = '%(asctime)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description='parametrs')
    parser.add_argument('--id', type=int, help='Id of camera', default=0)
    parser.add_argument('--res', type=str, help='Resolution', default='640x480')
    parser.add_argument('--fps', type=float, help='Frame per second', default=60)

    

    args = parser.parse_args()
    resolut = args.res.split('x')
    resol = []
    resol.append(int(resolut[0]))
    resol.append(int(resolut[1]))
    fps = args.fps
    cam = SensorCam(args.id, resol)
    queue0, queue1, queue2 = queue.Queue(10), queue.Queue(10), queue.Queue(10)
    sensor0, sensor1, sensor2 = SensorX(1), SensorX(0.1), SensorX(0.01)
    
    threading.Thread(target=sensor_loop, args=(sensor0, queue0), daemon=True).start()
    threading.Thread(target=sensor_loop, args=(sensor1, queue1), daemon=True).start()
    threading.Thread(target=sensor_loop, args=(sensor2, queue2), daemon=True).start()

    rate0, rate1, rate2 = 0, 0, 0
    org = (20,40)

    while True:
        frame = cam.get()
        
        if not queue0.empty():
            rate0 = queue0.get_nowait()

        if not queue1.empty():
            rate1 = queue1.get_nowait()

        if not queue2.empty():
            rate2 = queue2.get_nowait()

        frame = cv2.putText(frame, f'Сенсор 0: {rate0}', org, cv2.FONT_HERSHEY_COMPLEX, 1.2, (255,255,255), 1)
        frame = cv2.putText(frame, f'Сенсор 1: {rate1}', (org[0], org[1] + 25), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255,255,255), 1)
        frame = cv2.putText(frame, f'Сенсор 2: {rate2}', (org[0], org[1] + 50), cv2.FONT_HERSHEY_COMPLEX, 1.2, (255,255,255), 1)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1000 // fps) & 0xFF == 27:  
            break

    cam.__del__()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
