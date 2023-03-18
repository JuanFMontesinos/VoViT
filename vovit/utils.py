import os
import io
import re
import sys
import yaml
import json
import subprocess
from typing import Union
from collections import deque
from itertools import zip_longest

import numpy as np

VOVIT_LIB_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANDMARK_LIB_PATH = os.path.join(VOVIT_LIB_PATH, '3DDFA_V2')
DEFAULT_CFG_PATH = os.path.join(LANDMARK_LIB_PATH, 'configs', 'mb1_120x120.yml')

sys.path.append(LANDMARK_LIB_PATH)


class BaseDict(dict):
    def __add__(self, other):
        o_keys = other.keys()
        for key in self.keys():
            if key in o_keys:
                raise KeyError('Cannot concatenate both dictionaries. Key %s duplicated' % key)
        self.update(other)
        return self

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def write(self, path):
        path = os.path.splitext(path)[0]
        with open('%s.json' % path, 'w') as outfile:
            json.dump(self, outfile)

    def load(self, path):
        with open(path, 'r') as f:
            datastore = json.load(f)
            self.update(datastore)
        return self


def get_duration_fps(filename, display):
    """
    Wraps ffprobe to get file duration and fps

    :param filename: str, Path to file to be evaluate
    :param display: ['ms','s','min','h'] Time format miliseconds, sec, minutes, hours.
    :return: tuple(time, fps) in the mentioned format
    """

    def ffprobe2ms(time):
        cs = int(time[-2::])
        s = int(os.path.splitext(time[-5::])[0])
        idx = time.find(':')
        h = int(time[0:idx - 1])
        m = int(time[idx + 1:idx + 3])
        return [h, m, s, cs]

    # Get length of video with filename
    time = None
    fps = None
    result = subprocess.Popen(["ffprobe", str(filename)],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = [str(x) for x in result.stdout.readlines()]
    info_lines = [x for x in output if "Duration:" in x or "Stream" in x]
    duration_line = [x for x in info_lines if "Duration:" in x]
    fps_line = [x for x in info_lines if "Stream" in x]
    if duration_line:
        duration_str = duration_line[0].split(",")[0]
        pattern = '\d{2}:\d{2}:\d{2}.\d{2}'
        dt = re.findall(pattern, duration_str)[0]
        time = ffprobe2ms(dt)
    if fps_line:
        pattern = '(\d{2})(.\d{2})* fps'
        fps_elem = re.findall(pattern, fps_line[0])[0]
        fps = float(fps_elem[0] + fps_elem[1])
    if display == 's':
        time = time[0] * 3600 + time[1] * 60 + time[2] + time[3] / 100.0
    elif display == 'ms':
        time = (time[0] * 3600 + time[1] * 60 + time[2] + time[3] / 100.0) * 1000
    elif display == 'min':
        time = (time[0] * 3600 + time[1] * 60 + time[2] + time[3] / 100.0) / 60
    elif display == 'h':
        time = (time[0] * 3600 + time[1] * 60 + time[2] + time[3] / 100.0) / 3600
    return (time, fps)


def ffmpeg_call(video_path: str, dst_path: str, input_options: list, output_options: list, ext: None):
    """
    Runs ffmpeg for the following format for a single input/output:
        ffmpeg [input options] -i input [output options] output


    :param video_path: str Path to input video
    :param dst_path: str Path to output video
    :param input_options: List[str] list of ffmpeg options ready for a Popen format
    :param output_options: List[str] list of ffmpeg options ready for a Popen format
    :return: None
    """
    assert os.path.isfile(video_path)
    assert os.path.isdir(os.path.dirname(dst_path))
    if ext is not None:
        dst_path = os.path.splitext(dst_path)[0] + ext
    result = subprocess.Popen(["ffmpeg", *input_options, '-i', video_path, *output_options, dst_path],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout = result.stdout.read().decode("utf-8")
    stderr = result.stderr
    if stdout != '':
        print(stdout)
    if stderr is not None:
        print(stderr.read().decode("utf-8"))


def ffmpeg_join(video_path: str, audio_path: str, dst_path: str):
    """
    Runs ffmpeg for the following format for a single input/output:
        ffmpeg [input options] -i input [output options] output


    :param video_path: str Path to input video
    :param dst_path: str Path to output video
    :param input_options: List[str] list of ffmpeg options ready for a Popen format
    :param output_options: List[str] list of ffmpeg options ready for a Popen format
    :return: None
    """
    assert os.path.isfile(video_path)
    assert os.path.isfile(audio_path)
    assert os.path.isdir(os.path.dirname(dst_path))

    result = subprocess.Popen(["ffmpeg",
                               '-i', video_path, '-i', audio_path,
                               '-vcodec', 'copy', '-acodec', 'libmp3lame',
                               dst_path],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout = result.stdout.read().decode("utf-8")
    stderr = result.stderr
    if stdout != '':
        print(stdout)
    if stderr is not None:
        print(stderr.read().decode("utf-8"))


def np_int2float(waveform: np.ndarray, raise_error: bool = False) -> np.ndarray:
    """
    Cast an audio array in integer format into float scaling properly .
    :param waveform: numpy array of an audio waveform in int format
    :type waveform: np.ndarray
    :param raise_error: Flag to raise an error if dtype is not int
    """

    if waveform.dtype == np.int8:
        return (waveform / 128).astype(np.float32)

    elif waveform.dtype == np.int16:
        return (waveform / 32768).astype(np.float32)

    elif waveform.dtype == np.int32:
        return (waveform / 2147483648).astype(np.float32)
    elif waveform.dtype == np.int64:
        return (waveform / 9223372036854775808).astype(np.float32)
    elif raise_error:
        raise TypeError(f'int2float input should be of type np.intXX but {waveform.dtype} found')
    else:
        return waveform


def process_video(video_path,
                  video_dst=None,
                  landmarks_dst=None,
                  metadata_dst=None,
                  config_file=DEFAULT_CFG_PATH,
                  onnx=True,
                  n_pre=1, n_next=1, start_frame=-1, end_frame=-1,
                  landmark_type='2d_sparse',
                  assert_fps=None,
                  assert_frames=True):
    """

    :param video_path: str
    :param video_dst: (Optional) str path to save the video with landmarks
    :param landmarks_dst: (Optional) str path to save the landmarks as numpy arrays
    :param metadata_dst: (Optional) str path to save the metadata
    :param config_file: check core configs
    :param onnx: bool T o use onnx (recommended) or gpu
    :param smoothing: bool smooth videos temporally
    :param n_pre: int if smoothing, how many previous frames to sue
    :param n_next: int if  smoothing, how many post frames to use
    :param start_frame: int start processing from the n_th frame
    :param end_frame: int process until the m_th frame
    :param landmark_type: '2d_sparse', '2d_dense', '3d'
    :param assert_fps: int sets an assertion to ensure video fps
    :param assert_frames: bool verify duration*fps=number of real frames
    """
    assert landmark_type in ['2d_sparse', '2d_dense', '3d'], f'Landmarks should be either 2d_sparse, 2d_dense or 3d ' \
                                                             f'but {landmark_type} found.'
    save_video = isinstance(video_dst, str)
    save_landmarks = isinstance(landmarks_dst, str)
    save_metadata = isinstance(metadata_dst, str)
    if save_video:
        filename, ext = os.path.splitext(video_dst)
        if 'mp4' not in ext:
            print(f'only mp4 video format supported')
        video_dst = filename + '.mp4'
        basename = os.path.dirname(video_dst)
        assert os.path.exists(basename), f'Basename of video_dst {video_dst} does not exist'
    if save_landmarks:
        filename, ext = os.path.splitext(landmarks_dst)
        landmarks_dst = filename + '.npy'
        basename = os.path.dirname(landmarks_dst)
        assert os.path.exists(basename), f'Basename of video_dst {landmarks_dst} does not exist'
    if save_metadata:
        filename, ext = os.path.splitext(metadata_dst)
        landmarks_dst = filename + '.npy'
        basename = os.path.dirname(metadata_dst)
        assert os.path.exists(basename), f'Basename of video_dst {metadata_dst} does not exist'
    cfg = yaml.load(open(config_file), Loader=yaml.SafeLoader)
    cfg['checkpoint_fp'] = os.path.join(LANDMARK_LIB_PATH, 'weights', 'mb1_120x120.pth')
    cfg['bfm_fp'] = os.path.join(LANDMARK_LIB_PATH, 'configs', 'bfm_noneck_v3.pkl')

    from utils.functions import cv_draw_landmark
    from utils.render import render
    import imageio
    # Init FaceBoxes and TDDFA, recommend using onnx flag
    if onnx:
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['OMP_NUM_THREADS'] = '4'

        from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
        from TDDFA_ONNX import TDDFA_ONNX

        face_boxes = FaceBoxes_ONNX()
        tddfa = TDDFA_ONNX(**cfg)
    else:
        from TDDFA import TDDFA
        tddfa = TDDFA(gpu_mode='gpu', **cfg)
        face_boxes = FaceBoxes()
    duration, fps = get_duration_fps(video_path, 's')
    expected_frames = int(duration * fps)
    metadata = BaseDict()
    metadata['duration'] = duration
    metadata['fps'] = fps
    reader = imageio.get_reader(video_path)
    if assert_fps is not None:
        assert fps == assert_fps, f'FPS required to be {assert_fps} but video is {fps} FPS'
    if save_video:
        writer = imageio.get_writer(video_dst, fps=fps)

    # the simple implementation of average smoothing by looking ahead by n_next frames
    # assert the frames of the video >= n
    n_pre, n_next = n_pre, n_next
    n = n_pre + n_next + 1
    queue_ver = deque()
    queue_frame = deque()
    landmarks = []
    # run
    dense_flag = landmark_type in ('2d_dense', '3d',)
    pre_ver = None
    initial_frame = True
    for i, frame in enumerate(reader):

        if start_frame > 0 and i < start_frame:
            continue

        if end_frame > 0 and i > end_frame:
            break

        frame_bgr = frame[..., ::-1]  # RGB->BGR

        if initial_frame:
            initial_frame = False
            # detect
            boxes = face_boxes(frame_bgr)  # xmin, ymin, xmax, ymax, score
            if len(boxes) == 0:
                return True
            boxes = [boxes[0]]
            param_lst, roi_box_lst = tddfa(frame_bgr, boxes)
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]

            # refine
            param_lst, roi_box_lst = tddfa(frame_bgr, [ver], crop_policy='landmark')
            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
            for _ in range(n_pre):
                queue_ver.append(ver.copy())
            queue_ver.append(ver.copy())
            if save_video:
                for _ in range(n_pre):
                    queue_frame.append(frame_bgr.copy())
                queue_frame.append(frame_bgr.copy())

        else:
            param_lst, roi_box_lst = tddfa(frame_bgr, [pre_ver], crop_policy='landmark')

            roi_box = roi_box_lst[0]
            # todo: add confidence threshold to judge the tracking is failed
            if abs(roi_box[2] - roi_box[0]) * abs(roi_box[3] - roi_box[1]) < 2020:
                boxes = face_boxes(frame_bgr)

                boxes = [boxes[0]]

                param_lst, roi_box_lst = tddfa(frame_bgr, boxes)

            ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=dense_flag)[0]
            queue_ver.append(ver.copy())
            if save_video:
                queue_frame.append(frame_bgr.copy())

        pre_ver = ver  # for tracking

        # smoothing: enqueue and dequeue ops
        if len(queue_ver) >= n:
            ver_ave = np.mean(queue_ver, axis=0)

            landmarks.append(ver_ave)

            if save_video:
                if landmark_type == '2d_sparse':
                    img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave)  # since we use padding
                elif landmark_type == '2d_dense':
                    img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave, size=1)
                elif landmark_type == '3d':
                    img_draw = render(queue_frame[n_pre], [ver_ave], tddfa.tri, alpha=0.7)
                else:
                    raise ValueError(f'Unknown opt {landmark_type}')

                writer.append_data(img_draw[:, :, ::-1])  # BGR->RGB
                queue_frame.popleft()
            queue_ver.popleft()

    # we will lost the last n_next frames, still padding
    for _ in range(n_next):
        queue_ver.append(ver.copy())

        # the last frame

        ver_ave = np.mean(queue_ver, axis=0)
        landmarks.append(ver_ave)
        if save_video:
            queue_frame.append(frame_bgr.copy())
            if landmark_type == '2d_sparse':
                img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave)  # since we use padding
            elif landmark_type == '2d_dense':
                img_draw = cv_draw_landmark(queue_frame[n_pre], ver_ave, size=1)
            elif landmark_type == '3d':
                img_draw = render(queue_frame[n_pre], [ver_ave], tddfa.tri, alpha=0.7)
            else:
                raise ValueError(f'Unknown opt {landmark_type}')

            writer.append_data(img_draw[..., ::-1])  # BGR->RGB
            queue_frame.popleft()
        queue_ver.popleft()

    if assert_frames:
        if abs(expected_frames - i) > 5:
            print(f'A duration of {duration} at {fps} fps implies {expected_frames} frames ' \
                  f'but {i} frames found')
            return True
    if save_video:
        writer.close()
    if save_landmarks:
        np.save(landmarks_dst, np.stack(landmarks).round().astype(np.int16))
    return False


def landmarks2img(landmarks: np.ndarray, img: np.ndarray = None) -> np.ndarray:
    """
        Plot landmarks on a blank image.
        args:
            landmarks: NumPy array of shape (2,68)
            img: NumPy array of shape (H,W,3)
    """
    ml = landmarks[:2].transpose(1, 0)
    plt.axis('off')
    plt.tight_layout()
    fig = plt.figure(figsize=(5, 4))
    if img is not None:
        plt.imshow(img)
    plt.scatter(ml[:, 0], -ml[:, 1], alpha=0.8, color='red', s=20)  # 20
    plt.plot(ml[:17, 0], -ml[:17, 1], color='green', alpha=0.6)
    plt.plot(ml[17:22, 0], -ml[17:22, 1], color='green', alpha=0.6)
    plt.plot(ml[22:27, 0], -ml[22:27, 1], color='green', alpha=0.6)
    plt.plot(ml[27:31, 0], -ml[27:31, 1], color='green', alpha=0.6)
    plt.plot(ml[31:36, 0], -ml[31:36, 1], color='green', alpha=0.6)
    plt.plot(ml[36:42, 0], -ml[36:42, 1], color='green', alpha=0.6)
    plt.plot([ml[41, 0], ml[36, 0]], [-ml[41, 1], -ml[36, 1]], color='green', alpha=0.6)
    plt.plot(ml[42:48, 0], -ml[42:48, 1], color='green', alpha=0.6)
    plt.plot([ml[47, 0], ml[42, 0]], [-ml[47, 1], -ml[42, 1]], color='green', alpha=0.6)
    plt.plot(ml[48:60, 0], -ml[48:60, 1], color='green', alpha=0.6)
    plt.plot([ml[48, 0], ml[59, 0]], [-ml[48, 1], -ml[59, 1]], color='green', alpha=0.6)
    plt.plot(ml[60:, 0], -ml[60:, 1], color='green', alpha=0.6)
    plt.plot([ml[60, 0], ml[67, 0]], [-ml[60, 1], -ml[67, 1]], color='green', alpha=0.6)
    plt.axis('off')

    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')  # , dpi=DPI)
    io_buf.seek(0)
    img = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                     newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    plt.close(fig)
    return img


def landmarkseq2video(sequence: np.array, imgs: Union[np.array, list] = []) -> np.array:
    """
        Convert a sequence of landmarks to a video.
        args:
            sequence: NumPy array of shape (N,2,68)
            imgs: iterable of M (M<=N) NumPy arrays of shape (H,W,3)
    """
    for frame, img in zip_longest(sequence, imgs):
        yield landmarks2img(frame, img)
