# mobilenet_ssd_visual_live.py
# Live MobileNet-SSD detection + distance overlay and visuals
# Place this file with MobileNetSSD_deploy.prototxt and MobileNetSSD_deploy.caffemodel
# Run: python mobilenet_ssd_visual_live.py

import cv2, time, os, numpy as np
from collections import deque

# MODEL FILES (must be in same folder)
PROTO = "MobileNetSSD_deploy.prototxt"
MODEL = "MobileNetSSD_deploy.caffemodel"

# PASCAL VOC classes for MobileNet-SSD
CLASSES = ["background","person","aeroplane","bicycle","bird","boat",
           "bottle","bus","car","cat","Phones","chair","cow","diningtable",
           "motorbike","person","pottedplant",
           "sofa","train","tvmonitor"]

# Map tvmonitor -> laptop (if you want laptops detected)
LABEL_ALIAS = {"tvmonitor": "laptop"}

# Approx real heights (meters) — adjust as needed for better distance estimates
REAL_HEIGHTS = {
    "person": 1.70,
    "laptop": 0.04,
    "bottle": 0.25,
    "chair": 1.0,
    "sofa": 0.9,
    "car": 1.5
}

CONF_THRESH = 0.25   # detection confidence threshold
DEFAULT_FOCAL = 800.0  # starting focal length (px) — calibrate with 'r'
REFERENCE_POINT = None  # will be bottom-center of frame

# Simple tracker - assign short-term IDs by nearest-centroid matching
class SimpleTracker:
    def __init__(self, max_dist=200):
        self.next_id = 1
        self.objs = {}        # id -> (cx, cy, label)
        self.trails = {}      # id -> deque of points
        self.max_dist = max_dist

    def update(self, dets):
        # dets: list of (cx, cy, label)
        if len(self.objs) == 0:
            for (cx, cy, lab) in dets:
                oid = self.next_id; self.next_id += 1
                self.objs[oid] = (cx, cy, lab)
                self.trails[oid] = deque(maxlen=32)
                self.trails[oid].append((cx, cy))
            return self.objs
        # match by minimal euclidean distance
        assigned = {}
        used_new = set()
        for oid, (ox, oy, olab) in list(self.objs.items()):
            best = None; bestd = 1e9; bestdet = None
            for i, (cx, cy, lab) in enumerate(dets):
                if i in used_new: continue
                d = np.hypot(ox - cx, oy - cy)
                if d < bestd:
                    bestd = d; best = i; bestdet = (cx, cy, lab)
            if best is not None and bestd <= self.max_dist:
                used_new.add(best)
                assigned[oid] = bestdet
            else:
                # lost -> remove
                del self.objs[oid]; del self.trails[oid]
        # remaining new detections -> assign new ids
        for i, (cx, cy, lab) in enumerate(dets):
            if i in used_new: continue
            nid = self.next_id; self.next_id += 1
            assigned[nid] = (cx, cy, lab)
            self.trails[nid] = deque(maxlen=32)
        # update objects and trails
        self.objs = {oid: (c[0], c[1], c[2]) for oid,c in assigned.items()}
        for oid,(cx,cy,lab) in assigned.items():
            self.trails[oid].append((cx,cy))
        return self.objs

# distance estimate via pinhole model
def estimate_distance_m(real_h_m, focal_px, bbox_h_px):
    if bbox_h_px <= 0 or focal_px <= 0 or real_h_m <= 0:
        return None
    return (real_h_m * focal_px) / float(bbox_h_px)

def draw_label_box(img, text, topleft, bgcolor=(255,255,255), text_color=(0,0,0)):
    x,y = topleft
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5; th = 1
    (w,h), _ = cv2.getTextSize(text, font, scale, th)
    cv2.rectangle(img, (x,y), (x+w+6, y+h+6), bgcolor, -1)
    cv2.putText(img, text, (x+3, y+h), font, scale, text_color, th, cv2.LINE_AA)

def main():
    global REFERENCE_POINT
    if not os.path.exists(PROTO) or not os.path.exists(MODEL):
        print("Model files missing. Place prototxt and caffemodel here.")
        return

    print("[INFO] Loading model...")
    net = cv2.dnn.readNetFromCaffe(PROTO, MODEL)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("[INFO] Model loaded.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    REFERENCE_POINT = (W//2, H-10)  # bottom-center near frame bottom

    tracker = SimpleTracker(max_dist=220)
    focal_px = DEFAULT_FOCAL
    saving = False
    out_writer = None
    os.makedirs("runs", exist_ok=True)

    fps = 0.0; prev = time.time()
    show_help = True

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] frame read failed")
            break
        orig = frame.copy()
        h,w = frame.shape[:2]

        # draw blue border like in your screenshot
        border_th = 4
        cv2.rectangle(frame, (10,10), (w-10,h-10), (255,100,10), border_th)  # blue-ish (BGR order)
        # Note: colors BGR, above is orange-blue mix to approximate screenshot; change if needed.

        # DNN inference
        blob = cv2.dnn.blobFromImage(cv2.resize(orig, (300,300)), 0.007843, (300,300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        dets_for_track = []  # list of (cx, cy, label)
        boxes = []           # list of (bbox,label,conf)

        for i in range(detections.shape[2]):
            conf = float(detections[0,0,i,2])
            if conf < CONF_THRESH: continue
            idx = int(detections[0,0,i,1])
            label = CLASSES[idx] if idx < len(CLASSES) else str(idx)
            if label in LABEL_ALIAS:
                label = LABEL_ALIAS[label]
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (sX,sY,eX,eY) = box.astype("int")
            # sanitize
            sX, sY = max(0,sX), max(0,sY)
            eX, eY = min(w-1,eX), min(h-1,eY)
            cx, cy = (sX+eX)//2, (sY+eY)//2
            dets_for_track.append((cx, cy, label))
            boxes.append(((sX,sY,eX,eY), label, conf))

        # update tracker
        _ = tracker.update(dets_for_track)

        # draw each detection with cyan box for devices and big blue for person
        for (bbox,label,conf) in boxes:
            sX,sY,eX,eY = bbox
            box_h = eY - sY
            # choose color
            if label == "person":
                color = (255,0,0)   # blue (BGR)
            else:
                color = (255,200,0) # cyan-ish
            cv2.rectangle(frame, (sX,sY), (eX,eY), color, 2)
            # distance estimate
            rh = REAL_HEIGHTS.get(label, None)
            if rh is not None:
                dist = estimate_distance_m(rh, focal_px, box_h)
                if dist is not None:
                    txt = f"{label} {conf:.2f}"
                    cv2.putText(frame, txt, (sX, sY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    # distance badge
                    dist_text = f"{dist:.2f} m"
                    draw_label_box(frame, dist_text, (sX, eY+6), bgcolor=(255,255,255), text_color=(0,160,0))
                else:
                    cv2.putText(frame, f"{label} {conf:.2f}", (sX, sY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                cv2.putText(frame, f"{label} {conf:.2f}", (sX, sY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # draw tracker IDs + trails + magenta lines from reference point to centroids
        for oid, (cx,cy,label) in tracker.objs.items():
            # ID text
            cv2.putText(frame, f"ID:{oid}", (cx+6, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            # trail
            pts = tracker.trails.get(oid, [])
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], (255,0,255), 2)
            # magenta line from REFERENCE_POINT (bottom center) to centroid
            cv2.line(frame, REFERENCE_POINT, (cx,cy), (255,0,255), 2)
            # small green centroid dot
            cv2.circle(frame, (cx,cy), 4, (0,255,0), -1)

        # overlay counts and FPS and pixel/meter optionally
        counts = {}
        for _,_,lab in dets_for_track:
            counts[lab] = counts.get(lab, 0) + 1
        count_text = "Counts: " + ", ".join([f"{k}:{v}" for k,v in counts.items()]) if counts else "Counts: 0"
        cur = time.time()
        fps = 0.9*fps + 0.1*(1.0 / (cur - prev + 1e-6)) if 'fps' in locals() else 0.0
        prev = cur
        cv2.putText(frame, f"FPS: {fps:.1f} | Focal(px): {focal_px:.1f}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(frame, count_text, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,255), 2)

        # help overlay
        if show_help:
            overlay_y = 70
            cv2.putText(frame, "Controls: q=quit  s=toggle save  r=calibrate focal  +/- adjust focal  h=toggle help", (10,overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1)

        # show and optionally save
        cv2.imshow("visioneye-distance-calculation", frame)
        if saving:
            if out_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out_writer = cv2.VideoWriter(os.path.join("runs","visual_output.avi"), fourcc, 20.0, (w,h))
            out_writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            saving = not saving
            print("[INFO] saving toggled:", saving)
            if not saving and out_writer is not None:
                out_writer.release(); out_writer = None
        elif key == ord('h'):
            show_help = not show_help
        elif key in (ord('+'), ord('=')):
            focal_px *= 1.1
        elif key in (ord('-'), ord('_')):
            focal_px /= 1.1
        elif key == ord('r'):
            # interactive calibration: place a known object (e.g., person) at known distance and draw its bbox then press 'r'
            print("[CALIBRATE] Enter real object label (example: person, laptop) and distance in meters (e.g. 1.2)")
            lab = input("label: ").strip()
            try:
                dist_m = float(input("distance_m: ").strip())
            except:
                print("[CALIBRATE] invalid distance, abort")
                continue
            # find first bbox of that label in current boxes
            found = None
            for (sX,sY,eX,eY), l, conf in boxes:
                if l == lab:
                    found = (sX,sY,eX,eY); break
            if found is None:
                print("[CALIBRATE] no bbox of that label found on current frame")
            else:
                b_h = found[3] - found[1]
                if b_h <= 0:
                    print("[CALIBRATE] invalid bbox height")
                else:
                    focal_px = (b_h * dist_m) / REAL_HEIGHTS.get(lab, 1.0)
                    print(f"[CALIBRATE] computed focal_px = {focal_px:.2f}")
        # end key handling

    cap.release()
    if out_writer is not None:
        out_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
