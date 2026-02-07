import sys
import csv
import datetime
import time
import numpy as np
from collections import deque

from scipy.stats import gamma
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QFormLayout, QGroupBox, QLabel, QLineEdit,
                             QComboBox, QDoubleSpinBox, QSpinBox, QPushButton, QCheckBox, QMessageBox)
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg

# --- 1. RADAR PHYSICS ---
class RadarEnvironment:
    def __init__(self, num_bins=1200):
        self.num_bins = num_bins
        self.range_axis = np.linspace(0, 50, num_bins)

    def generate_clutter_k_dist(self, shape_param=1.0, scale_param=1.0):
            # 1. Texture (Gamma distribution: modeling the 'surges' in the sea)
            # Often denoted by the 'nu' parameter in K-distribution literature
            texture = np.random.gamma(shape=shape_param, scale=scale_param, size=self.num_bins)

            # 2. Speckle (Complex Gaussian / Rayleigh component)
            u_real = np.random.normal(0, np.sqrt(0.5), self.num_bins)
            u_imag = np.random.normal(0, np.sqrt(0.5), self.num_bins)
            speckle = np.abs(u_real + 1j * u_imag)**2

            # 3. Compound result (Modulated Speckle)
            clutter = texture * speckle

            # Apply Range Loss
            range_loss = 1.0 / (np.maximum(self.range_axis, 1.0)**2.5)
            range_loss = range_loss / range_loss[0]
            return clutter * range_loss

    def generate_clutter(self, shape_param=1.0, scale_param=5.0):
        # Texture & Speckle
        texture = gamma.rvs(shape_param, scale=scale_param, size=self.num_bins)
        u = np.random.randn(self.num_bins) + 1j * np.random.randn(self.num_bins)
        speckle_power = (np.abs(u)**2) / 2.0
        clutter = texture * speckle_power

        # Range Loss (1/R^2.5 for visualization purposes)
        range_loss = 1.0 / (np.maximum(self.range_axis, 1.0)**2.5)
        range_loss = range_loss / range_loss[0]
        return clutter * range_loss

    def generate_target_rcs(self, swerling_type, mean_snr_db):
        mean_rcs = 10**(mean_snr_db/10.0)
        # Thermal noise floor offset
        mean_rcs = mean_rcs * 0.1

        if swerling_type == "Swerling 0 (Steady)":
            return mean_rcs
        elif swerling_type == "Swerling 1 (Slow)":
            return np.random.exponential(mean_rcs)
        elif swerling_type == "Swerling 2 (Fast)":
            return np.mean(np.random.exponential(mean_rcs, size=10))
        elif swerling_type == "Swerling 3 (Dom + Small)":
            return np.random.chisquare(4) * (mean_rcs / 4.0)
        elif swerling_type == "Swerling 4 (Fast Dom)":
            return np.mean(np.random.chisquare(4, size=10) * (mean_rcs / 4.0))
        return mean_rcs

# --- 2. CFAR ALGORITHMS (Refactored for C++ Porting) ---

def cfar_ca(signal, num_train, num_guard, pfa):
    """Cell Averaging CFAR: Best for homogeneous noise."""
    num_cells = len(signal)
    thresholds = np.zeros(num_cells)

    # Alpha Calculation (C++ equivalent: std::pow)
    N = 2 * num_train
    alpha = N * (pfa**(-1/N) - 1)

    for i in range(num_cells):
        # Window Logic (C++ equivalent: loops or iterators)
        start = max(0, i - num_train - num_guard)
        end = min(num_cells, i + num_train + num_guard + 1)
        cut_start = max(0, i - num_guard)
        cut_end = min(num_cells, i + num_guard + 1)

        left = signal[start:cut_start]
        right = signal[cut_end:end]

        if len(left) > 0 or len(right) > 0:
            # CA Logic: Average of all reference cells
            # C++: Sum loop over left, Sum loop over right, divide by count
            mean_noise = (np.sum(left) + np.sum(right)) / (len(left) + len(right))
            thresholds[i] = mean_noise * alpha
        else:
            thresholds[i] = 1e6

    return thresholds

def cfar_go(signal, num_train, num_guard, pfa):
    """Greatest Of CFAR: Best for clutter edges."""
    num_cells = len(signal)
    thresholds = np.zeros(num_cells)

    N = 2 * num_train
    alpha = N * (pfa**(-1/N) - 1)

    for i in range(num_cells):
        start = max(0, i - num_train - num_guard)
        end = min(num_cells, i + num_train + num_guard + 1)
        cut_start = max(0, i - num_guard)
        cut_end = min(num_cells, i + num_guard + 1)

        left = signal[start:cut_start]
        right = signal[cut_end:end]

        if len(left) > 0 or len(right) > 0:
            # GO Logic: Max of Left Mean vs Right Mean
            l_mean = np.mean(left) if len(left) > 0 else 0
            r_mean = np.mean(right) if len(right) > 0 else 0
            thresholds[i] = max(l_mean, r_mean) * alpha
        else:
            thresholds[i] = 1e6
    return thresholds

def cfar_so(signal, num_train, num_guard, pfa):
    """Smallest Of CFAR: Best for multi-target."""
    num_cells = len(signal)
    thresholds = np.zeros(num_cells)

    N = 2 * num_train
    alpha = N * (pfa**(-1/N) - 1)

    for i in range(num_cells):
        start = max(0, i - num_train - num_guard)
        end = min(num_cells, i + num_train + num_guard + 1)
        cut_start = max(0, i - num_guard)
        cut_end = min(num_cells, i + num_guard + 1)

        left = signal[start:cut_start]
        right = signal[cut_end:end]

        if len(left) > 0 or len(right) > 0:
            # SO Logic: Min of Left Mean vs Right Mean
            l_mean = np.mean(left) if len(left) > 0 else 1e9
            r_mean = np.mean(right) if len(right) > 0 else 1e9
            thresholds[i] = min(l_mean, r_mean) * alpha
        else:
            thresholds[i] = 1e6
    return thresholds

def cfar_os(signal, num_train, num_guard, pfa):
    """Ordered Statistic CFAR: Robust in clutter."""
    num_cells = len(signal)
    thresholds = np.zeros(num_cells)

    # OS Alpha approximation
    alpha = -np.log(pfa)

    for i in range(num_cells):
        start = max(0, i - num_train - num_guard)
        end = min(num_cells, i + num_train + num_guard + 1)
        cut_start = max(0, i - num_guard)
        cut_end = min(num_cells, i + num_guard + 1)

        left = signal[start:cut_start]
        right = signal[cut_end:end]

        # Combine explicitly for sorting
        # C++: You would fill a std::vector and use std::nth_element
        ref = np.concatenate([left, right])

        if len(ref) > 0:
            k = int(len(ref) * 0.75)
            idx = min(k, len(ref)-1)
            # Efficient partial sort (faster than full sort)
            noise = np.partition(ref, idx)[idx]
            thresholds[i] = noise * alpha
        else:
            thresholds[i] = 1e6

    return thresholds

# --- Main Dispatcher Function ---
def run_cfar(signal, algo_type, num_train, num_guard, pfa):
    """
    Switchboard that routes the GUI request to the specific implementation.
    """
    if algo_type == "CA-CFAR":
        return cfar_ca(signal, num_train, num_guard, pfa)
    elif algo_type == "GO-CFAR":
        return cfar_go(signal, num_train, num_guard, pfa)
    elif algo_type == "SO-CFAR":
        return cfar_so(signal, num_train, num_guard, pfa)
    elif algo_type == "OS-CFAR":
        return cfar_os(signal, num_train, num_guard, pfa)
    else:
        # Fallback
        return np.zeros_like(signal)

# --- 3. GUI ---
class RadarSimulatorPG(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Radar Simulator with Data Logger")
        self.resize(1300, 900)
        print("Hello")

        self.env = RadarEnvironment(num_bins=1200)
        self.target_indices = [300, 600, 950]

        # Metric History (Rolling Buffer)
        self.history_len = 50
        self.stats_history = deque(maxlen=self.history_len) # Stores tuples: (TP, FN, FP)

        # Logging State
        self.is_logging = False
        self.log_start_time = 0
        self.log_buffer = [] # Stores rows for CSV

        # UI
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        print("Hello 1")

        # --- LEFT PANEL (Controls) ---
        left_layout = QVBoxLayout()
        layout.addLayout(left_layout, stretch=0)

        # 1. Params Group
        panel_params = QGroupBox("Radar Parameters")
        panel_params.setFixedWidth(280)
        form = QFormLayout()

        print("Hello 3")

        self.combo_algo = QComboBox()
        self.combo_algo.addItems(["CA-CFAR", "GO-CFAR", "SO-CFAR", "OS-CFAR"])

        self.spin_guard = QSpinBox()
        self.spin_guard.setRange(0, 50)  # Allow 0 to 50 guard cells
        self.spin_guard.setValue(2)      # Default to 2
        self.spin_guard.setToolTip("Number of guard cells on one side")

        self.spin_pfa = QDoubleSpinBox()
        self.spin_pfa.setRange(1.0, 9.0); self.spin_pfa.setValue(4.0)

        self.combo_clutter_model = QComboBox()
        self.combo_clutter_model.addItems(["Standard (Gamma)", "Compound K-Dist"])

        self.spin_sea = QDoubleSpinBox()
        self.spin_sea.setRange(0.1, 10.0); self.spin_sea.setValue(0.5); self.spin_sea.setSingleStep(0.1)
        self.spin_sea.setToolTip("Lower value (0.1) = Spikier Waves")

        self.spin_clutter_scale = QDoubleSpinBox()
        self.spin_clutter_scale.setRange(0.1, 20.0); self.spin_clutter_scale.setValue(5.0)

        self.spin_dead_zone = QDoubleSpinBox()
        self.spin_dead_zone.setRange(0.0, 10.0); self.spin_dead_zone.setValue(2.0)
        self.spin_dead_zone.setSuffix(" km")

        self.spin_snr = QDoubleSpinBox()
        self.spin_snr.setRange(0.0, 50.0); self.spin_snr.setValue(15.0)
        self.spin_snr.setSuffix(" dB")

        self.combo_swerling = QComboBox()
        self.combo_swerling.addItems([ "Swerling 4 (Fast Dom)", "Swerling 3 (Big Ship)", "Swerling 2 (Fast)", "Swerling 1 (Slow)", "Swerling 0 (Steady)"])

        self.spin_noise_floor = QDoubleSpinBox()
        self.spin_noise_floor.setRange(0.001, 2.0)
        self.spin_noise_floor.setValue(0.1)
        self.spin_noise_floor.setSingleStep(0.01)
        self.spin_noise_floor.setToolTip("Receiver Thermal Noise Power")

        self.edit_target_ranges = QLineEdit("12.5, 25.0, 39.5") # Default ranges in km
        self.edit_target_ranges.setToolTip("Enter target ranges in km, separated by commas")


        self.chk_auto = QCheckBox("Run Realtime"); self.chk_auto.setChecked(True)
        self.chk_auto.stateChanged.connect(self.toggle_timer)

        form.addRow("Algorithm:", self.combo_algo)
        form.addRow("Guard Cells:", self.spin_guard)
        form.addRow("PFA (1e-X):", self.spin_pfa)
        form.addRow("Clutter Model:", self.combo_clutter_model)
        form.addRow("Sea K-Shape:", self.spin_sea)
        form.addRow("Sea Scale:", self.spin_clutter_scale)
        form.addRow("Receiver Noise:", self.spin_noise_floor)

        form.addRow("Blind Zone:", self.spin_dead_zone)
        form.addRow("Target Ranges (km):", self.edit_target_ranges)
        form.addRow("Target SNR:", self.spin_snr)
        form.addRow("Target Type:", self.combo_swerling)
        form.addRow(self.chk_auto)
        panel_params.setLayout(form)
        left_layout.addWidget(panel_params)

        # 2. Metrics Group
        panel_metrics = QGroupBox("Performance Metrics")
        panel_metrics.setFixedWidth(280)
        metrics_form = QFormLayout()

        self.spin_window = QSpinBox()
        self.spin_window.setRange(10, 500); self.spin_window.setValue(50)
        self.spin_window.valueChanged.connect(self.update_window_size)

        print("Hello 8")

        self.lbl_pd = QLabel("0.0 %")
        self.lbl_pfa = QLabel("0.0")
        self.lbl_tp = QLabel("0")
        self.lbl_fp = QLabel("0")
        self.lbl_fn = QLabel("0")

        self.lbl_pd.setStyleSheet("font-weight: bold; color: green; font-size: 14px;")
        self.lbl_fp.setStyleSheet("font-weight: bold; color: red; font-size: 14px;")

        metrics_form.addRow("Avg Window (N):", self.spin_window)
        metrics_form.addRow("Prob Detection (Pd):", self.lbl_pd)
        metrics_form.addRow("Prob False Alarm:", self.lbl_pfa)
        metrics_form.addRow("--- Counts (Last N) ---", QLabel(""))
        metrics_form.addRow("Hits (TP):", self.lbl_tp)
        metrics_form.addRow("Misses (FN):", self.lbl_fn)
        metrics_form.addRow("False Alarms (FP):", self.lbl_fp)

        panel_metrics.setLayout(metrics_form)
        left_layout.addWidget(panel_metrics)

        print("Hello 8")

        # 3. Data Logger Group (NEW)
        panel_log = QGroupBox("Data Logger")
        panel_log.setFixedWidth(280)
        log_layout = QVBoxLayout()

        log_form = QFormLayout()
        self.spin_log_duration = QSpinBox()
        self.spin_log_duration.setRange(1, 300)
        self.spin_log_duration.setValue(10)
        self.spin_log_duration.setSuffix(" sec")

        print("Hello 8")

        log_form.addRow("Duration:", self.spin_log_duration)

        self.btn_log = QPushButton("Start Logging to CSV")
        self.btn_log.clicked.connect(self.start_logging)
        self.btn_log.setStyleSheet("background-color: #E0E0E0; font-weight: bold; padding: 5px;")

        # Inside the 'Data Logger Group' layout
        self.btn_batch = QPushButton("Run SNR Sweep (Pd vs SNR)")
        self.btn_batch.clicked.connect(self.run_batch_test)
        self.btn_batch.setStyleSheet("background-color: #D1E8FF; font-weight: bold; padding: 5px;")

        print("Hello 9")

        log_layout.addLayout(log_form)
        log_layout.addWidget(self.btn_log)
        panel_log.setLayout(log_layout)
        left_layout.addWidget(panel_log)
        log_layout.addWidget(self.btn_batch)

        left_layout.addStretch()

        # --- RIGHT PANEL (Plots) ---
        plot_layout = QVBoxLayout()
        pg.setConfigOption('background', 'k')
        pg.setConfigOption('foreground', 'd')

        print("Hello 10")

        self.plot_scope = pg.PlotWidget(title="A-Scope (Log Power)")
        print("Hello 10.1")
        self.plot_scope.showGrid(x=True, y=True, alpha=0.3)
        print("Hello 10.2")
        self.plot_scope.setLogMode(x=False, y=True)
        self.plot_scope.setYRange(-1, 3)
        self.plot_scope.setLabel('bottom', 'Range', units='km')
        self.plot_scope.addLegend(offset=(10, 10))

        print("Hello 10.3")

        self.curve_signal = self.plot_scope.plot(pen=pg.mkPen('#00FF00', width=1), name='Raw Signal')
        self.curve_thresh = self.plot_scope.plot(pen=pg.mkPen('#FF0000', width=2, style=Qt.DashLine), name='Threshold')
        self.scatter_targets = self.plot_scope.plot(pen=None, symbol='o', symbolBrush='b', symbolSize=10, name='Target Truth')

        print("Hello 10.4")
        self.plot_binary = pg.PlotWidget(title="Binary Detection Stream")
        self.plot_binary.setYRange(-0.1, 1.1)
        self.plot_binary.setLabel('bottom', 'Range', units='km')
        self.curve_detect = self.plot_binary.plot(pen=pg.mkPen('#FFFF00', width=2), fillLevel=0, brush=(255,255,0,50))

        print("Hello 10.5")
        plot_layout.addWidget(self.plot_scope)
        plot_layout.addWidget(self.plot_binary)
        layout.addLayout(plot_layout, stretch=1)

        print("Hello 11")

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(50)

        print("Hello End")

    def toggle_timer(self, state):
        if state == Qt.Checked: self.timer.start(50)
        else: self.timer.stop()

    def update_window_size(self):
        new_len = self.spin_window.value()
        self.stats_history = deque(list(self.stats_history), maxlen=new_len)

    def start_logging(self):
        if self.is_logging: return

        self.is_logging = True
        self.log_start_time = time.time()
        self.log_buffer = [] # Clear buffer

        self.btn_log.setText("Logging... (Please Wait)")
        self.btn_log.setStyleSheet("background-color: #FFCCCC; color: red; font-weight: bold; padding: 5px;")
        self.btn_log.setEnabled(False)

    def save_log_to_file(self):
        # Generate filename with timestamp
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"radar_log_{ts}.csv"

        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                # Header
                writer.writerow(["Timestamp", "Algo", "Noise_Floor", "PFA_Set", "SNR_dB", "Sea_Shape", "Sea_Scale", "Blind_Zone_km", "Pd_Percent", "Pfa_Measured", "TP", "FP", "FN"])
                writer.writerows(self.log_buffer)

            print(f"Log saved to {filename}")
            # Optional: Show alert
            # QMessageBox.information(self, "Log Saved", f"Data saved to:\n{filename}")

        except Exception as e:
            print(f"Error saving log: {e}")

        # Reset UI
        self.is_logging = False
        self.btn_log.setText("Start Logging to CSV")
        self.btn_log.setStyleSheet("background-color: #E0E0E0; font-weight: bold; padding: 5px;")
        self.btn_log.setEnabled(True)


    def update_loop(self):
        # 1. Inputs
        sea_shape = self.spin_sea.value()
        sea_scale = self.spin_clutter_scale.value()
        clutter_model = self.combo_clutter_model.currentText()
        pfa_exp = self.spin_pfa.value()
        pfa = 10 ** (-pfa_exp)
        algo = self.combo_algo.currentText()
        swerling = self.combo_swerling.currentText()
        noise_power = self.spin_noise_floor.value()
        target_snr = self.spin_snr.value()
        dead_zone_km = self.spin_dead_zone.value()
        num_guard_cells = self.spin_guard.value()

        # 2. Physics
        if clutter_model == "Compound K-Dist":
            signal = self.env.generate_clutter_k_dist(shape_param=sea_shape, scale_param=sea_scale)
        else:
            signal = self.env.generate_clutter(shape_param=sea_shape, scale_param=sea_scale)

        signal += np.random.exponential(noise_power, size=len(signal))

        x_axis = self.env.range_axis
        target_x, target_y = [], []

        for idx in self.target_indices:
            rcs = self.env.generate_target_rcs(swerling, target_snr)
            signal[idx] += rcs
            target_x.append(x_axis[idx])
            target_y.append(signal[idx])

        try:
            raw_ranges = self.edit_target_ranges.text().split(',')
            # Convert km to bin index: index = (range / max_range) * num_bins
            dynamic_indices = []
            for r in raw_ranges:
                km = float(r.strip())
                idx = int((km / 50.0) * self.env.num_bins)
                if 0 <= idx < self.env.num_bins:
                    dynamic_indices.append(idx)
            self.target_indices = dynamic_indices # Update the active indices
        except ValueError:
            pass # Handle malformed input gracefully

        # 3. CFAR Detection
        threshold = run_cfar(signal, algo, num_train=15, num_guard=num_guard_cells, pfa=pfa)

        # Apply Blind Zone
        bins_per_km = self.env.num_bins / 50.0
        dead_zone_bins = int(dead_zone_km * bins_per_km)
        if dead_zone_bins > 0:
            threshold[0:dead_zone_bins] = 1e6

        detections = (signal > threshold).astype(int)

        # 4. SCORING (Robust Tolerance Version)
        tp_count = 0
        used_detection_indices = set()

        # A. Check Hits (+/- 2 bins)
        for t_idx in self.target_indices:
            # Skip targets inside blind zone
            if t_idx < dead_zone_bins: continue

            search_window = range(max(0, t_idx-2), min(len(detections), t_idx+3))

            hit_found = False
            for d_idx in search_window:
                if detections[d_idx] == 1:
                    hit_found = True
                    used_detection_indices.add(d_idx)

            if hit_found: tp_count += 1

        # B. Check Misses
        visible_targets = [t for t in self.target_indices if t >= dead_zone_bins]
        fn_count = len(visible_targets) - tp_count

        # C. Check False Alarms
        total_detections_indices = np.where(detections == 1)[0]
        fp_count = 0
        for d_idx in total_detections_indices:
            if d_idx >= dead_zone_bins and d_idx not in used_detection_indices:
                fp_count += 1

        # D. Stats Calculation (For UI and Logging)
        self.stats_history.append((tp_count, fn_count, fp_count))

        avg_tp = sum(x[0] for x in self.stats_history)
        avg_fn = sum(x[1] for x in self.stats_history)
        avg_fp = sum(x[2] for x in self.stats_history)

        total_targets = avg_tp + avg_fn
        if total_targets > 0: pd = (avg_tp / total_targets) * 100.0
        else: pd = 0.0

        total_opportunities = (len(signal) * len(self.stats_history)) - total_targets
        if total_opportunities > 0: pfa_measured = avg_fp / total_opportunities
        else: pfa_measured = 0.0

        # Update UI Labels
        self.lbl_tp.setText(str(avg_tp))
        self.lbl_fn.setText(str(avg_fn))
        self.lbl_fp.setText(str(avg_fp))
        self.lbl_pd.setText(f"{pd:.1f} %")
        self.lbl_pfa.setText(f"{pfa_measured:.5f}")

        # 5. DATA LOGGING (If Active)
        if self.is_logging:
            # Append current frame stats to buffer
            current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            row = [
                current_time, algo, noise_power, pfa_exp, target_snr, sea_shape, sea_scale, dead_zone_km,
                pd, pfa_measured, avg_tp, avg_fp, avg_fn
            ]
            self.log_buffer.append(row)

            # Check duration
            elapsed = time.time() - self.log_start_time
            if elapsed >= self.spin_log_duration.value():
                self.save_log_to_file()

        # 6. Plotting
        self.curve_signal.setData(x_axis, signal)
        self.curve_thresh.setData(x_axis, threshold)
        self.scatter_targets.setData(target_x, target_y)
        self.curve_detect.setData(x_axis, detections)

    def run_batch_test(self):
        """Sweeps SNR from 0 to 30 dB and calculates Pd for each algorithm."""
        # 1. Setup Sweep Parameters
        snr_range = np.arange(0, 31, 2) # 0, 2, 4 ... 30 dB
        iterations_per_snr = 100 # How many frames to average
        algos_to_test = ["CA-CFAR", "GO-CFAR", "SO-CFAR", "OS-CFAR"]

        # Data storage: { 'CA-CFAR': [pd1, pd2, ...], ... }
        results = {algo: [] for algo in algos_to_test}

        # 2. Get current environment settings from GUI
        sea_shape = self.spin_sea.value()
        sea_scale = self.spin_clutter_scale.value()
        clutter_model = self.combo_clutter_model.currentText()
        noise_power = self.spin_noise_floor.value()
        pfa = 10 ** (-self.spin_pfa.value())
        num_guard = self.spin_guard.value()

        # 3. Perform the Sweep
        # We disable the timer so the GUI doesn't conflict
        self.timer.stop()
        self.btn_batch.setText("Running Sweep...")
        QApplication.processEvents() # Refresh UI

        for algo in algos_to_test:
            for snr in snr_range:
                tp_total = 0
                targets_total = 0

                for _ in range(iterations_per_snr):
                    # Generate Physics
                    if clutter_model == "Compound K-Dist":
                        sig = self.env.generate_clutter_k_dist(sea_shape, sea_scale)
                    else:
                        sig = self.env.generate_clutter(sea_shape, sea_scale)

                    sig += np.random.exponential(noise_power, size=len(sig))

                    # Add Targets at dynamic positions
                    for idx in self.target_indices:
                        rcs = self.env.generate_target_rcs(self.combo_swerling.currentText(), snr)
                        sig[idx] += rcs

                    # Run CFAR
                    thresh = run_cfar(sig, algo, 15, num_guard, pfa)
                    dets = (sig > thresh).astype(int)

                    # Simple Scoring for Batch
                    for t_idx in self.target_indices:
                        targets_total += 1
                        # Check +/- 2 bin window
                        if np.any(dets[max(0, t_idx-2):min(len(dets), t_idx+3)] == 1):
                            tp_total += 1

                pd = (tp_total / targets_total) * 100
                results[algo].append(pd)

        # 4. Plotting the results
        self.show_roc_plot(snr_range, results)

        # Cleanup
        self.btn_batch.setText("Run SNR Sweep (Pd vs SNR)")
        self.timer.start(50)

    def show_roc_plot(self, snr_range, results):
            self.roc_win = pg.GraphicsLayoutWidget(title="Batch Test Results")
            self.roc_win.resize(800, 600)
            plot = self.roc_win.addPlot(title="Performance: Pd vs. SNR")
            plot.setLabel('bottom', "Target SNR", units='dB')
            plot.setLabel('left', "Probability of Detection (Pd)", units='%')
            plot.addLegend()
            plot.showGrid(x=True, y=True)

            colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00']
            for i, (algo, pds) in enumerate(results.items()):
                plot.plot(snr_range, pds, pen=pg.mkPen(colors[i], width=3), name=algo, symbol='o')

            self.roc_win.show()

def main():
    app = QApplication(sys.argv)
    print(app)

    # 3. Create and show the Window
    window = RadarSimulatorPG()
    print(window)
    window.show()

    # 4. Start the Event Loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
