from collections import defaultdict, deque
import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st 
import supervision as sv

SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])

TARGET_WIDTH = 25
TARGET_HEIGHT = 250

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1],
    ]
)


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)



track_thresh = st.sidebar.slider("Tracking Threshold : " , min_value=0.1 , max_value=1.0 , value=0.3)
detections_confidence = st.sidebar.slider("Detections Confidence : " , min_value=0.1 , max_value=1.0 , value=0.5)
nms_threshold = st.number_input("NMS Threshold :", min_value=0.1 , max_value=1.0 , value=0.6)
source_path = st.text_input("Entre your video path :")

if source_path : 
    if st.button("Click To start") : 
        frame_window = st.image( [] )
        video_info = sv.VideoInfo.from_video_path(video_path=source_path)
        model = YOLO("yolov8n.pt")
        byte_track = sv.ByteTrack(
            frame_rate=video_info.fps, track_thresh=track_thresh
        )
        thickness = sv.calculate_dynamic_line_thickness(
            resolution_wh=video_info.resolution_wh
        )
        text_scale = sv.calculate_dynamic_text_scale(resolution_wh=video_info.resolution_wh)
        bounding_box_annotator = sv.EllipseAnnotator(thickness=thickness)
        label_annotator = sv.LabelAnnotator(
            text_scale=text_scale,
            text_thickness=thickness,
            text_position=sv.Position.BOTTOM_CENTER,
        )
        trace_annotator = sv.TraceAnnotator(
            thickness=thickness,
            trace_length=video_info.fps * 2,
            position=sv.Position.BOTTOM_CENTER,
        )
        frame_generator = sv.get_video_frames_generator(source_path=source_path)
        polygon_zone = sv.PolygonZone(
            polygon=SOURCE, frame_resolution_wh=video_info.resolution_wh
        )
        view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
        coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))


        for frame in frame_generator:
            result = model(frame)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[detections.confidence > detections_confidence]
            detections = detections[polygon_zone.trigger(detections)]
            detections = detections.with_nms(threshold=nms_threshold)
            detections = byte_track.update_with_detections(detections=detections)
            points = detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
            )
            points = view_transformer.transform_points(points=points).astype(int)
            for tracker_id, [_, y] in zip(detections.tracker_id, points):
                coordinates[tracker_id].append(y)
            labels = []
            for tracker_id in detections.tracker_id:
                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    labels.append(f"#{tracker_id}")
                else:
                    coordinate_start = coordinates[tracker_id][-1]
                    coordinate_end = coordinates[tracker_id][0]
                    distance = abs(coordinate_start - coordinate_end)
                    time = len(coordinates[tracker_id]) / video_info.fps
                    speed = distance / time * 3.6
                    labels.append(f"#{tracker_id} {int(speed)} km/h")
            annotated_frame = frame.copy()
            annotated_frame = trace_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = bounding_box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )
            annotated_frame = cv2.cvtColor( annotated_frame , cv2.COLOR_BGR2RGB)
            frame_window.image(annotated_frame)