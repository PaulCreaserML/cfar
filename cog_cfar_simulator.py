import sys
import numpy as np
import pyqtgraph as pg
from scipy.stats import gamma
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QFormLayout, QGroupBox, QLabel,
                             QComboBox, QCheckBox)
from PyQt5.QtCore import Qt, QTimer

# --- 1. CFAR ALGORITHMS (Isolated Functions) ---
def cfar_ca(signal, num_train, num_guard, pfa):
    num_cells = len(signal)
    thresholds = np.full(num_cells, 1e6)
    N = 2 * num_train
    # Protection against zero/negative PFA
    safe_pfa = max(pfa, 1e-10)
    alpha = N * (safe_pfa**(-1/N) - 1)

    for i in range(num_cells):
        start, end = max(0, i-num_train-num_guard), min(num_cells, i+num_train+num_guard+1)
        cut_start, cut_end = max(0, i-num_guard), min(num_cells, i+num_guard+1)

        left = signal[start:cut_start]
        right = signal[cut_end:end]

        if len(left) > 0 or len(right) > 0:
            # Simple averaging
            mean_val = (np.sum(left) + np.sum(right)) / (len(left) + len(right))
            thresholds[i] = mean_val * alpha
    return thresholds

def cfar_os(signal, num_train, num_guard, pfa):
    num_cells = len(signal)
    thresholds = np.full(num_cells, 1e6)
    safe_pfa = max(pfa, 1e-10)
    alpha = -np.log(safe_pfa)

    for i in range(num_cells):
        start, end = max(0, i-num_train-num_guard), min(num_cells, i+num_train+num_guard+1)
        cut_start, cut_end = max(0, i-num_guard), min(num_cells, i+num_guard+1)

        ref = np.concatenate([signal[start:cut_start], signal[cut_end:end]])

        if len(ref) > 0:
            # 3/4 rank OS
            k = int(len(ref) * 0.75)
            # Fast partition sort
            idx = min(k, len(ref)-1)
            noise = np.partition(ref, idx)[idx]
            thresholds[i] = noise * alpha
    return thresholds

# --- 2. INTELLIGENCE LAYER (The "Brain") ---
class CognitiveProcessor:
    def __init__(self):
        self.current_algo = "CA-CFAR"
        self.variability_index = 0.0


# ... inside CognitiveProcessor class ...

    def analyze_environment(self, signal):
        """
        Calculates VI (Variability Index) smartly.
        It filters out the 'Thermal Noise' so we only measure the Sea State.
        """
        # 1. Look at the Near Range (first ~10km / 250 bins)
        region = signal[0:250]

        # 2. Estimate the Noise Floor (using the quietest 10% of samples)
        # This gives us a baseline for "Silence"
        noise_floor = np.percentile(region, 10)

        # 3. Filter: Only analyze bins that are ACTIVE (e.g., > 2x Noise Floor)
        # This removes the "flat line" data that was diluting your stats.
        active_clutter = region[region > (2.0 * noise_floor)]

        # Safety check: If sea is dead calm, active_clutter might be empty
        if len(active_clutter) < 10:
            # Fallback: Just use the raw region if everything is quiet
            analysis_data = region
        else:
            analysis_data = active_clutter

        # 4. Calculate VI on the filtered data
        mean_val = np.mean(analysis_data)
        std_val = np.std(analysis_data)

        if mean_val > 0:
            vi = 1.0 + (std_val**2) / (mean_val**2)
        else:
            vi = 1.0

        self.variability_index = vi
        return vi

    def select_best_algo(self):
        # VI Threshold for switching
        # Normal Rayleigh noise has VI around 1.2 - 1.8
        # Heavy K-Dist clutter usually exceeds 2.5 or 3.0
        if self.variability_index > 8.0:
            self.current_algo = "OS-CFAR"
            reason = "High Clutter Variability (Spiky)"
        else:
            self.current_algo = "CA-CFAR"
            reason = "Homogeneous Background (Smooth)"

        return self.current_algo, reason

# --- 3. SCENARIO PARAMETERS ---
SHIP_TYPES = {
    "Small Fishing Boat": {"snr_db": 12.0, "swerling": "Swerling 1"},
    "Patrol Boat":        {"snr_db": 18.0, "swerling": "Swerling 1"},
    "Large Tanker":       {"snr_db": 30.0, "swerling": "Swerling 3"},
    "Stealth Target":     {"snr_db": 8.0,  "swerling": "Swerling 1"}
}

# Tweaked values to make visual difference obvious
WEATHER_CONDITIONS = {
    "Calm Sea":    {"shape": 20.0, "scale": 1.0},   # Very Smooth, Weak
    "Choppy Water": {"shape": 1.5,  "scale": 10.0}, # Moderately Spiky
    "Heavy Storm":  {"shape": 0.1,  "scale": 60.0}  # <--- MASSIVE BOOST in Power
}

# --- 4. MAIN SIMULATOR ---
class CognitiveRadarSim(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cognitive Radar Simulator (Scenario Based)")
        self.resize(1200, 800)

        self.brain = CognitiveProcessor()
        self.target_indices = [200, 500, 900]
        self.range_axis = np.linspace(0, 50, 1200)

        # UI Setup
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Left Panel
        left_layout = QVBoxLayout()

        # Scenario Box
        panel_scenario = QGroupBox("Scenario Setup")
        form = QFormLayout()
        self.combo_ship = QComboBox()
        self.combo_ship.addItems(SHIP_TYPES.keys())
        self.combo_weather = QComboBox()
        self.combo_weather.addItems(WEATHER_CONDITIONS.keys())
        form.addRow("Target Profile:", self.combo_ship)
        form.addRow("Weather Condition:", self.combo_weather)
        panel_scenario.setLayout(form)
        left_layout.addWidget(panel_scenario)

        # Brain Box
        panel_brain = QGroupBox("Cognitive Processor")
        brain_layout = QVBoxLayout()
        self.chk_auto = QCheckBox("Enable Auto-Selection (Cognitive Mode)")
        self.chk_auto.setChecked(True) # Default to ON
        self.chk_auto.setStyleSheet("font-weight: bold; color: blue;")

        self.lbl_vi = QLabel("Env Variability: 0.0")
        self.lbl_decision = QLabel("Decision: Manual")
        self.lbl_algo_active = QLabel("Active: CA-CFAR")
        self.lbl_algo_active.setStyleSheet("font-size: 16px; font-weight: bold; color: green;")

        brain_layout.addWidget(self.chk_auto)
        brain_layout.addWidget(self.lbl_vi)
        brain_layout.addWidget(self.lbl_decision)
        brain_layout.addWidget(self.lbl_algo_active)
        panel_brain.setLayout(brain_layout)
        left_layout.addWidget(panel_brain)

        left_layout.addStretch()
        layout.addLayout(left_layout, stretch=0)

        # Right Panel (Plot)
        self.plot_scope = pg.PlotWidget(title="A-Scope")
        self.plot_scope.addLegend()
        self.plot_scope.setLogMode(x=False, y=True)
        self.plot_scope.setYRange(-1, 3.5) # Tweaked Y range
        self.plot_scope.setLabel('bottom', "Range", units='km')

        self.curve_signal = self.plot_scope.plot(pen='g', name='Signal')
        self.curve_thresh = self.plot_scope.plot(pen=pg.mkPen('r', width=2, style=Qt.DashLine), name='Threshold')
        layout.addWidget(self.plot_scope, stretch=1)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(50)

    def generate_signal(self, weather_key, ship_key):
        # 1. Physics: Weather (Clutter)
        w_params = WEATHER_CONDITIONS[weather_key]
        shape, scale = w_params['shape'], w_params['scale']

        # Generate Gamma distributed texture
        texture = gamma.rvs(shape, scale=scale, size=1200)
        # Generate Speckle (Rayleigh Power)
        u = np.random.randn(1200) + 1j * np.random.randn(1200)
        speckle = np.abs(u)**2 / 2.0

        clutter = texture * speckle

        # Apply Range Loss (Make it visible in first 15km)
        # Using 1/R^2.2 to keep clutter visible longer
        range_loss = 1.0 / (np.maximum(self.range_axis, 1.0)**2.2)
        signal = clutter * (range_loss / range_loss[0])

        # Add Thermal Noise Floor
        signal += np.random.exponential(0.1, size=1200)

        # 2. Physics: Ships
        s_params = SHIP_TYPES[ship_key]
        snr_db = s_params['snr_db']
        swerling = s_params['swerling']

        # Convert dB to Linear (relative to noise floor 0.1)
        mean_rcs = (10**(snr_db/10.0)) * 0.1

        for idx in self.target_indices:
            val = mean_rcs
            if "Swerling 1" in swerling:
                val = np.random.exponential(mean_rcs)
            elif "Swerling 3" in swerling:
                val = np.random.chisquare(4) * (mean_rcs/4.0)

            signal[idx] += val

        return signal

    def update_loop(self):
        # 1. Get Settings
        weather = self.combo_weather.currentText()
        ship = self.combo_ship.currentText()

        # 2. Generate
        signal = self.generate_signal(weather, ship)

        # 3. Brain Processing
        vi = self.brain.analyze_environment(signal)
        self.lbl_vi.setText(f"Env Variability Index (Near): {vi:.2f}")

        if self.chk_auto.isChecked():
            algo_name, reason = self.brain.select_best_algo()
            self.lbl_decision.setText(f"Decision: {reason}")
            self.lbl_algo_active.setText(f"Active: {algo_name}")

            # Color coding the label for feedback
            if algo_name == "OS-CFAR":
                self.lbl_algo_active.setStyleSheet("font-size: 16px; font-weight: bold; color: red;")
            else:
                self.lbl_algo_active.setStyleSheet("font-size: 16px; font-weight: bold; color: green;")
        else:
            algo_name = "CA-CFAR" # Manual Default
            self.lbl_decision.setText("Decision: Manual Override")
            self.lbl_algo_active.setText(f"Active: {algo_name}")
            self.lbl_algo_active.setStyleSheet("font-size: 16px; font-weight: bold; color: grey;")

        # 4. Execute CFAR
        if algo_name == "CA-CFAR":
            thresh = cfar_ca(signal, 15, 2, 1e-4)
        else:
            thresh = cfar_os(signal, 15, 2, 1e-4)

        # 5. Plot
        self.curve_signal.setData(self.range_axis, signal)
        self.curve_thresh.setData(self.range_axis, thresh)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CognitiveRadarSim()
    window.show()
    sys.exit(app.exec_())
