import sys
import os
import cv2
import math
from PyQt5 import QtCore, QtGui, QtWidgets

# stereo_clipper.py
# GUI tool to load two matching videos, preview one, mark zones to remove on a timeline,
# and export clipped copies of both videos with the same time chunks removed.


def seconds_to_str(s: float) -> str:
    s = max(0.0, float(s))
    m, sec = divmod(int(round(s)), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"

def merge_intervals(intervals, eps=1e-3):
    if not intervals:
        return []
    intervals = sorted((min(a,b), max(a,b)) for a,b in intervals)
    merged = [intervals[0]]
    for a,b in intervals[1:]:
        la, lb = merged[-1]
        if a <= lb + eps:  # overlap or touching
            merged[-1] = (la, max(lb, b))
        else:
            merged.append((a,b))
    return merged

class TimelineWidget(QtWidgets.QWidget):
    positionChanged = QtCore.pyqtSignal(float)
    zonesChanged = QtCore.pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(48)
        self.setMouseTracking(True)
        self.duration = 0.0
        self.position = 0.0
        self.zones = []  # list of (start_sec, end_sec)

        # Drag-to-create-zone state
        self.dragging = False
        self.drag_start_sec = None
        self.drag_curr_sec = None

        # Spacebar highlight mode state (press space to start/stop)
        self.highlight_active = False
        self.highlight_start_sec = None
        self.highlight_curr_sec = None

        self.setFocusPolicy(QtCore.Qt.StrongFocus)

    def setDuration(self, seconds: float):
        self.duration = max(0.0, float(seconds))
        self.position = min(self.position, self.duration)
        self.update()

    def setPosition(self, seconds: float):
        self.position = max(0.0, min(float(seconds), self.duration))
        self.update()

    def setZones(self, zones):
        self.zones = merge_intervals(zones)
        self.update()
        self.zonesChanged.emit(self.zones)

    def clearZones(self):
        self.zones = []
        self.update()
        self.zonesChanged.emit(self.zones)

    def seconds_at_x(self, x: int) -> float:
        w = max(1, self.width())
        ratio = max(0.0, min(1.0, x / w))
        return ratio * self.duration

    def x_at_seconds(self, s: float) -> int:
        if self.duration <= 0:
            return 0
        ratio = max(0.0, min(1.0, s / self.duration))
        return int(ratio * self.width())

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if self.duration <= 0:
            return
        sec = self.seconds_at_x(e.x())
        if e.button() == QtCore.Qt.LeftButton:
            # If in highlight mode, treat as scrub (no new zone drag)
            if self.highlight_active:
                self.setPosition(sec)
                self.highlight_curr_sec = sec
                self.positionChanged.emit(sec)
                self.update()
                return
            # Start zone drag
            self.dragging = True
            self.drag_start_sec = sec
            self.drag_curr_sec = sec
            # Also scrub immediately
            self.setPosition(sec)
            self.positionChanged.emit(sec)
            self.update()
        elif e.button() == QtCore.Qt.RightButton:
            # Right-click: remove nearest zone if clicked inside it
            removed = False
            new = []
            for (a, b) in self.zones:
                if a - 0.02 <= sec <= b + 0.02:
                    removed = True
                    continue
                new.append((a, b))
            if removed:
                self.setZones(new)

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if self.duration <= 0:
            return
        sec = self.seconds_at_x(e.x())
        if self.dragging:
            # Update drag preview and scrub while dragging
            self.drag_curr_sec = sec
            self.setPosition(sec)
            self.positionChanged.emit(sec)
            self.update()
        elif self.highlight_active:
            # Update highlight end while hovering; also scrub
            self.highlight_curr_sec = sec
            self.setPosition(sec)
            self.positionChanged.emit(sec)
            self.update()
        else:
            # Free scrubbing by hovering/dragging over the timeline
            if e.buttons() & QtCore.Qt.LeftButton or True:
                self.setPosition(sec)
                self.positionChanged.emit(sec)

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        if self.duration <= 0:
            return
        if e.button() == QtCore.Qt.LeftButton and self.dragging:
            self.dragging = False
            a = self.drag_start_sec
            b = self.drag_curr_sec
            self.drag_start_sec = None
            self.drag_curr_sec = None
            if a is not None and b is not None and abs(b - a) >= 0.02:
                self.setZones(self.zones + [(min(a, b), max(a, b))])
            else:
                # Treat as click to set position
                sec = self.seconds_at_x(e.x())
                self.setPosition(sec)
                self.positionChanged.emit(sec)
            self.update()

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if self.duration <= 0:
            return
        step = max(0.01, self.duration / 200.0)

        if e.key() == QtCore.Qt.Key_Space:
            # Toggle highlight execution: start/end a zone at the current playhead
            e.accept()  # prevent window-level space shortcut from toggling play
            if not self.highlight_active:
                # Start highlight
                self.highlight_active = True
                self.highlight_start_sec = self.position
                self.highlight_curr_sec = self.position
                self.update()
            else:
                # End highlight and commit zone
                self.highlight_active = False
                a = self.highlight_start_sec
                b = self.highlight_curr_sec if self.highlight_curr_sec is not None else self.position
                self.highlight_start_sec = None
                self.highlight_curr_sec = None
                if a is not None and b is not None and abs(b - a) >= 0.02:
                    self.setZones(self.zones + [(min(a, b), max(a, b))])
                self.update()
            return
        elif e.key() in (QtCore.Qt.Key_Left,):
            self.setPosition(self.position - step)
            self.positionChanged.emit(self.position)
        elif e.key() in (QtCore.Qt.Key_Right,):
            self.setPosition(self.position + step)
            self.positionChanged.emit(self.position)
        elif e.key() == QtCore.Qt.Key_Backspace:
            if self.zones:
                self.setZones(self.zones[:-1])
        else:
            super().keyPressEvent(e)

    def paintEvent(self, e: QtGui.QPaintEvent):
        p = QtGui.QPainter(self)
        rect = self.rect()
        p.fillRect(rect, QtGui.QColor(35, 35, 35))
        # Colors
        zone_color = QtGui.QColor(200, 60, 60, 180)
        line_color = QtGui.QColor(60, 160, 220)
        highlight_color = QtGui.QColor(255, 200, 50, 120)
        bar_rect = rect.adjusted(4, 8, -4, -8)
        p.fillRect(bar_rect, QtGui.QColor(55, 55, 55))
        # total bar outline
        pen = QtGui.QPen(QtGui.QColor(90, 90, 90))
        pen.setWidth(1)
        p.setPen(pen)
        p.drawRect(bar_rect)
        # zones
        for a, b in self.zones:
            x1 = self.x_at_seconds(a)
            x2 = self.x_at_seconds(b)
            zr = QtCore.QRect(min(x1, x2), bar_rect.top(), abs(x2 - x1), bar_rect.height())
            p.fillRect(zr, zone_color)
        # dragging preview
        if self.dragging and self.drag_start_sec is not None and self.drag_curr_sec is not None:
            x1 = self.x_at_seconds(self.drag_start_sec)
            x2 = self.x_at_seconds(self.drag_curr_sec)
            zr = QtCore.QRect(min(x1, x2), bar_rect.top(), abs(x2 - x1), bar_rect.height())
            p.fillRect(zr, highlight_color)
        # highlight mode preview
        if self.highlight_active and self.highlight_start_sec is not None:
            end_sec = self.highlight_curr_sec if self.highlight_curr_sec is not None else self.position
            x1 = self.x_at_seconds(self.highlight_start_sec)
            x2 = self.x_at_seconds(end_sec)
            zr = QtCore.QRect(min(x1, x2), bar_rect.top(), abs(x2 - x1), bar_rect.height())
            p.fillRect(zr, highlight_color)
        # playhead
        x = self.x_at_seconds(self.position)
        p.setPen(QtGui.QPen(line_color, 2))
        p.drawLine(x, bar_rect.top() - 4, x, bar_rect.bottom() + 4)
        # time labels
        p.setPen(QtGui.QColor(220, 220, 220))
        p.drawText(rect.adjusted(6, 0, -6, -rect.height() + 16), QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter, seconds_to_str(self.position))
        p.drawText(rect.adjusted(6, 0, -6, -rect.height() + 16), QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter, seconds_to_str(self.duration))
        p.end()

class ClipWorker(QtCore.QThread):
    progress = QtCore.pyqtSignal(str)
    done = QtCore.pyqtSignal(bool, str)

    def __init__(self, video1_path, video2_path, remove_zones, out1_path, out2_path):
        super().__init__()
        self.video1_path = video1_path
        self.video2_path = video2_path
        self.remove_zones = merge_intervals([(max(0.0,a), max(0.0,b)) for a,b in remove_zones])
        self.out1_path = out1_path
        self.out2_path = out2_path

    def run(self):
        try:
            self._clip_video_pair()
            self.done.emit(True, "Finished")
        except Exception as e:
            self.done.emit(False, f"Error: {e}")

    def _clip_video_pair(self):
        def compute_keep(duration, zones):
            z = merge_intervals([(max(0.0,a), min(duration,b)) for a,b in zones if a < duration and b > 0.0])
            keep = []
            cur = 0.0
            for (a,b) in z:
                if a > cur:
                    keep.append((cur, a))
                cur = max(cur, b)
            if cur < duration:
                keep.append((cur, duration))
            # filter near-zero
            keep = [(a,b) for a,b in keep if b - a > 1e-3]
            return keep

        def clip_one(src, dst, zones):
            cap = cv2.VideoCapture(src)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open {src}")
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else None
            duration = frame_count / fps if frame_count else self._estimate_duration(cap, fps)
            keep = compute_keep(duration, zones)
            if not keep:
                # create an empty short black video (or raise)
                self.progress.emit(f"No content to keep for {os.path.basename(src)}; skipping write.")
                cap.release()
                return
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            tmp_dst = dst
            writer = cv2.VideoWriter(tmp_dst, fourcc, fps, (width, height))
            if not writer.isOpened():
                cap.release()
                raise RuntimeError(f"Cannot open writer for {dst}")
            total_kept_frames = 0
            for (a,b) in keep:
                start_idx = max(0, int(math.floor(a * fps)))
                end_idx = max(start_idx, int(math.ceil(b * fps)))  # exclusive
                self.progress.emit(f"{os.path.basename(src)}: keeping {seconds_to_str(a)} - {seconds_to_str(b)}")
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
                idx = start_idx
                while idx < end_idx:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    writer.write(frame)
                    idx += 1
                    total_kept_frames += 1
            writer.release()
            cap.release()
            self.progress.emit(f"Wrote {total_kept_frames} frames to {os.path.basename(dst)}")

        # Determine reference duration from video1 for zones; but compute keep per-video
        # so both videos use same time seconds.
        self.progress.emit("Processing video 1...")
        clip_one(self.video1_path, self.out1_path, self.remove_zones)
        self.progress.emit("Processing video 2...")
        clip_one(self.video2_path, self.out2_path, self.remove_zones)

    def _estimate_duration(self, cap, fps):
        # fallback if CAP_PROP_FRAME_COUNT is missing
        # Try to binary seek to find end; otherwise assume 0
        pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
        end_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        if end_pos and end_pos > 0:
            return end_pos / fps
        return 0.0

class VideoPlayer(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stereo Clipper - Select zones to remove, then press Enter to export")
        self.video1_path = None
        self.video2_path = None
        self.cap = None
        self.fps = 30.0
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_timer)
        self.playing = False
        self.total_frames = 0
        self.duration = 0.0

        # UI
        self.video_label = QtWidgets.QLabel("Load a video pair to begin.")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(640, 360)
        self.video_label.setStyleSheet("background:#111; color:#999;")

        self.timeline = TimelineWidget()
        self.timeline.positionChanged.connect(self.on_timeline_position_changed)
        self.timeline.zonesChanged.connect(self.on_zones_changed)

        # Single "Load Videos" button + hidden dummy second button to keep wiring intact
        self.btn_load1 = QtWidgets.QPushButton("Load Video Pair")
        self.btn_load2 = QtWidgets.QPushButton()
        self.btn_load2.setVisible(False)
        self.btn_load2.setEnabled(False)
        self.btn_load2.setMaximumWidth(0)

        # Override both old handlers to a single loader that enforces selecting two files
        def _load_both():
            paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Select Two Videos", "", "Videos (*.mp4 *.mov *.mkv *.avi)"
            )
            if len(paths) != 2:
                QtWidgets.QMessageBox.warning(self, "Selection error", "Please select exactly two videos.")
                return
            self.video1_path, self.video2_path = paths[0], paths[1]
            self._open_preview(self.video1_path)
            self.lbl_status.setText(
            f"Loaded pair: {os.path.basename(self.video1_path)} + {os.path.basename(self.video2_path)}"
            )

        self.load_video1 = _load_both
        self.load_video2 = _load_both
        self.btn_play = QtWidgets.QPushButton("Play")
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_clear = QtWidgets.QPushButton("Clear Zones")
        self.btn_undo = QtWidgets.QPushButton("Undo Zone")
        self.btn_save = QtWidgets.QPushButton("Save (Enter)")

        self.lbl_info = QtWidgets.QLabel("Drag on the timeline to mark zones to REMOVE (highlighted). Right-click a zone to remove it. Space=Play/Pause. Enter=Save.")
        self.lbl_info.setStyleSheet("color:#ccc;")
        self.lbl_status = QtWidgets.QLabel("")
        self.lbl_status.setStyleSheet("color:#8fc;")

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self.btn_load1)
        controls.addWidget(self.btn_load2)
        controls.addStretch(1)
        controls.addWidget(self.btn_play)
        controls.addWidget(self.btn_pause)
        controls.addWidget(self.btn_undo)
        controls.addWidget(self.btn_clear)
        controls.addWidget(self.btn_save)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.video_label)
        layout.addWidget(self.timeline)
        layout.addLayout(controls)
        layout.addWidget(self.lbl_info)
        layout.addWidget(self.lbl_status)

        # Signals
        self.btn_load1.clicked.connect(self.load_video1)
        self.btn_load2.clicked.connect(self.load_video2)
        self.btn_play.clicked.connect(self.play)
        self.btn_pause.clicked.connect(self.pause)
        self.btn_clear.clicked.connect(self.timeline.clearZones)
        self.btn_undo.clicked.connect(self.undo_zone)
        self.btn_save.clicked.connect(self.save_clipped)

        # Shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Space), self, activated=self.toggle_play)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Return), self, activated=self.save_clipped)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Enter), self, activated=self.save_clipped)

        self.remove_zones = []

    def undo_zone(self):
        if self.remove_zones:
            self.remove_zones = self.remove_zones[:-1]
            self.timeline.setZones(self.remove_zones)

    def load_video1(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Video 1", "", "Videos (*.mp4 *.mov *.mkv *.avi)")
        if path:
            self.video1_path = path
            self._open_preview(path)

    def load_video2(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Video 2", "", "Videos (*.mp4 *.mov *.mkv *.avi)")
        if path:
            self.video2_path = path
            self.lbl_status.setText(f"Video 2 loaded: {os.path.basename(path)}")

    def _open_preview(self, path):
        if self.cap:
            self.cap.release()
            self.cap = None
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QtWidgets.QMessageBox.critical(self, "Error", f"Cannot open {path}")
            return
        self.cap = cap
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 0 else 30.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.get(cv2.CAP_PROP_FRAME_COUNT) > 0 else 0
        self.total_frames = frames
        self.duration = frames / self.fps if frames > 0 else 0.0
        if self.duration <= 0:
            # try approximate
            cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
            approx_frames = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.total_frames = approx_frames
            self.duration = approx_frames / self.fps if approx_frames > 0 else 0.0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.timeline.setDuration(self.duration)
        self.timeline.setPosition(0.0)
        self.remove_zones = []
        self.timeline.setZones([])
        self._read_and_show_frame(0)
        self.lbl_status.setText(f"Loaded: {os.path.basename(path)} | {seconds_to_str(self.duration)} @ {self.fps:.2f} fps")

    def _on_timer(self):
        if not self.cap:
            return
        ok, frame = self.cap.read()
        if not ok:
            self.pause()
            return
        cur_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        cur_sec = cur_idx / self.fps
        self.timeline.setPosition(cur_sec)
        self._show_frame(frame)
        if self.total_frames and cur_idx >= self.total_frames - 1:
            self.pause()

    def _read_and_show_frame(self, frame_index: int):
        if not self.cap:
            return
        frame_index = max(0, frame_index)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.cap.read()
        if ok:
            self._show_frame(frame)

    def _show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pix.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def resizeEvent(self, e: QtGui.QResizeEvent):
        super().resizeEvent(e)
        # Rescale last frame on resize
        # No-op; next frame will rescale. Optionally force repaint.

    def play(self):
        if not self.cap:
            return
        self.playing = True
        interval = int(1000 / max(1.0, self.fps))
        self.timer.start(interval)

    def pause(self):
        self.playing = False
        self.timer.stop()

    def toggle_play(self):
        if self.playing:
            self.pause()
        else:
            self.play()

    def on_timeline_position_changed(self, sec: float):
        if not self.cap or self.fps <= 0:
            return
        idx = int(max(0, min(self.total_frames-1 if self.total_frames else 10**9, round(sec * self.fps))))
        self._read_and_show_frame(idx)

    def on_zones_changed(self, zones):
        self.remove_zones = zones
        if not self.duration:
            return
        # Status summary
        keep = self._compute_keep(self.duration, zones)
        kept = sum((b-a) for a,b in keep)
        removed = sum((b-a) for a,b in merge_intervals(zones))
        self.lbl_status.setText(f"Zones to remove: {len(zones)} | Remove {removed:.2f}s | Keep {kept:.2f}s")

    def _compute_keep(self, duration, zones):
        z = merge_intervals([(max(0.0,a), min(duration,b)) for a,b in zones if a < duration and b > 0.0])
        keep = []
        cur = 0.0
        for (a,b) in z:
            if a > cur:
                keep.append((cur, a))
            cur = max(cur, b)
        if cur < duration:
            keep.append((cur, duration))
        return [(a,b) for a,b in keep if b-a > 1e-3]

    def save_clipped(self):
        if not self.video1_path or not self.video2_path:
            QtWidgets.QMessageBox.warning(self, "Missing videos", "Please load both Video 1 and Video 2.")
            return
        zones = merge_intervals(self.remove_zones)
        if not zones:
            resp = QtWidgets.QMessageBox.question(self, "No zones selected", "No remove zones selected. Save full copies?")
            if resp != QtWidgets.QMessageBox.Yes:
                return
        base1, ext1 = os.path.splitext(self.video1_path)
        base2, ext2 = os.path.splitext(self.video2_path)
        out1 = base1 + "_clipped.mp4"
        out2 = base2 + "_clipped.mp4"
        # Worker thread
        self.btn_save.setEnabled(False)
        self.btn_save.setText("Saving...")
        self.lbl_status.setText("Exporting...")
        self.worker = ClipWorker(self.video1_path, self.video2_path, zones, out1, out2)
        self.worker.progress.connect(self.on_worker_progress)
        self.worker.done.connect(self.on_worker_done)
        self.worker.start()

    def on_worker_progress(self, msg: str):
        self.lbl_status.setText(msg)

    def on_worker_done(self, ok: bool, msg: str):
        self.btn_save.setEnabled(True)
        self.btn_save.setText("Save (Enter)")
        self.lbl_status.setText(msg)
        if ok:
            QtWidgets.QMessageBox.information(self, "Done", "Clipped videos saved next to originals with _clipped suffix.")
        else:
            QtWidgets.QMessageBox.critical(self, "Error", msg)
        self.worker = None

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = VideoPlayer()
    w.resize(960, 600)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()