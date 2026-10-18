# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Qt window of the inlet explorer: the 3D view in the centre, the section sketcher docked on the left and a
compact control panel on the right.

The window only translates between widgets and ``explore_model.InletModel``; all state, history and rules
live in the model. The 3D view (pyvistaqt ``QtInteractor``) draws no widgets of its own.
"""
import html
import math
import sys

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets

from ductmgr import vspio
from ductmgr.explore_model import MIN_ALIGN_PROJECTION, MODE_PLACING, InletModel, Readout
from ductmgr.geometry import InletPlane, sweep_for_edge_angle
from ductmgr.qt_sketch import SketchDock
from ductmgr.viz import ACCENT, BACKGROUND, Scene, SceneView

# ---- UI constants (not design inputs) -------------------------------------------------------------
WINDOW_SIZE = (1560, 900)
PANEL_WIDTH = 350
SKETCH_WIDTH = 520
FONT_POINT_SIZE = 9
READOUT_FONT = "Consolas"
READOUT_FONT_POINT_SIZE = 8
READOUT_MIN_HEIGHT = 300
LABEL_WIDTH = 21  # characters, readout label column
NUDGE_STEPS = (0.001, 0.005, 0.01, 0.05, 0.1)  # m
DEFAULT_NUDGE_STEP = 0.01
POSITION_MARGIN = 0.05  # of the fuselage size, beyond its bounds, for the position spin ranges
PITCH_RANGE_DEG = (-85.0, 85.0)
SWEEP_RANGE_DEG = (-90.0, 90.0)
CLICK_TOLERANCE_PX = 4  # a press-release further apart than this is a camera drag, not a click
MANUAL_ANGLE_DEFAULT_DEG = 30.0  # the edge picker's starting manual angle
COLORS = {"info": "#C9D1D9", "pass": "#4CAF50", "warn": "#E0A030", "error": "#F85149"}  # readout text colours

STYLESHEET = f"""
QWidget {{ background: {BACKGROUND}; color: #D7DCE0; font-size: {FONT_POINT_SIZE}pt; }}
QMainWindow::separator {{ background: #2C333B; width: 1px; height: 1px; }}
QDockWidget {{ border: 0; }}
QDockWidget::title {{ background: #1B1F24; padding: 3px 6px; border-bottom: 1px solid #2C333B; }}
QScrollArea {{ border: 0; }}
QToolBar {{ background: #1B1F24; border: 0; border-bottom: 1px solid #2C333B; spacing: 3px; padding: 2px; }}
QToolButton {{ background: transparent; border: 1px solid transparent; border-radius: 3px; padding: 2px 7px; }}
QToolButton:hover {{ border-color: #2C333B; background: #1E2329; }}
QToolButton:checked {{ background: #1E2329; border-color: {ACCENT}; color: {ACCENT}; }}
QToolButton:disabled {{ color: #5A636C; }}
QDoubleSpinBox, QComboBox {{ background: #1E2329; border: 1px solid #2C333B; border-radius: 3px;
    padding: 1px 4px; min-height: 18px; selection-background-color: {ACCENT}; selection-color: #0B0E11; }}
QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {ACCENT}; }}
QDoubleSpinBox:disabled, QComboBox:disabled {{ color: #5A636C; background: #181C20; }}
QComboBox QAbstractItemView {{ background: #1E2329; selection-background-color: {ACCENT};
    selection-color: #0B0E11; border: 1px solid #2C333B; }}
QPushButton {{ background: #1E2329; border: 1px solid #2C333B; border-radius: 3px; padding: 3px 9px; }}
QPushButton:hover {{ border-color: {ACCENT}; }}
QPushButton:checked {{ background: {ACCENT}; color: #0B0E11; border-color: {ACCENT}; }}
QPushButton:disabled {{ color: #5A636C; border-color: #22272D; }}
QCheckBox::indicator {{ width: 13px; height: 13px; border: 1px solid #2C333B; border-radius: 2px;
    background: #1E2329; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QTextEdit {{ background: #101316; border: 1px solid #2C333B; border-radius: 3px; }}
QStatusBar {{ background: #1B1F24; color: #9AA4AD; border-top: 1px solid #2C333B; }}
QStatusBar QLabel {{ background: transparent; }}
QToolButton#groupHeader {{ font-weight: bold; color: {ACCENT}; text-align: left; border: 0;
    border-bottom: 1px solid #2C333B; padding: 3px 2px; }}
QToolTip {{ background: #1E2329; color: #D7DCE0; border: 1px solid {ACCENT}; }}
"""


def readouts_html(readouts: list[Readout]) -> str:
    """Monospace HTML for the read-only panel: PASS green, WARN amber, ERROR red, the rest neutral."""
    lines = []
    for r in readouts:
        text = f"{r.label:<{LABEL_WIDTH}} {r.value}" if r.label else r.value
        lines.append(f'<span style="color:{COLORS[r.level]}">{html.escape(text)}</span>')
    return '<pre style="margin:0; white-space:pre-wrap">' + "\n".join(lines) + "</pre>"


class FocusSpinBox(QtWidgets.QDoubleSpinBox):
    """A spin box that takes the mouse wheel only while it has keyboard focus (so scrolling the panel is safe)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setKeyboardTracking(False)  # typed values apply on Enter / focus loss, arrows and wheel at once

    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class CollapsibleGroup(QtWidgets.QWidget):
    """A titled section whose body can be folded away."""

    def __init__(self, title: str, expanded: bool = True, parent=None):
        super().__init__(parent)
        self.header = QtWidgets.QToolButton()
        self.header.setObjectName("groupHeader")
        self.header.setText(title)
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.header.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.body = QtWidgets.QWidget()
        self.form = QtWidgets.QFormLayout(self.body)
        self.form.setContentsMargins(6, 4, 2, 6)
        self.form.setVerticalSpacing(3)
        self.form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.header)
        layout.addWidget(self.body)
        self.header.toggled.connect(self._toggled)
        self._toggled(expanded)

    def _toggled(self, expanded: bool) -> None:
        self.body.setVisible(expanded)
        self.header.setArrowType(QtCore.Qt.ArrowType.DownArrow if expanded else QtCore.Qt.ArrowType.RightArrow)

    @property
    def expanded(self) -> bool:
        return self.header.isChecked()


def wing_display_names(wings) -> list[str]:
    """Wing names for the picker; a repeated name gets its running number (several geoms may share a name)."""
    totals = {}
    for w in wings:
        totals[w.name] = totals.get(w.name, 0) + 1
    seen = {}
    out = []
    for w in wings:
        seen[w.name] = seen.get(w.name, 0) + 1
        out.append(f"{w.name} #{seen[w.name]}" if totals[w.name] > 1 else w.name)
    return out


class AlignEdgeDialog(QtWidgets.QDialog):
    """Picks the edge to align the opening's planform trace with: a wing's leading / trailing edge (read from its
    VSP parameters in the section spanning the inlet's y) or a manual angle."""

    def __init__(self, wings, y: float, pitch_deg: float, parent=None):
        super().__init__(parent)
        self.wings, self.y, self.pitch_deg = list(wings), float(y), float(pitch_deg)
        self.setWindowTitle("Align to edge")
        self.radio_wing = QtWidgets.QRadioButton("Wing edge (from the VSP parameters)")
        self.combo_wing = QtWidgets.QComboBox()
        self.combo_wing.addItems(wing_display_names(self.wings))
        self.radio_le = QtWidgets.QRadioButton("Leading edge")
        self.radio_te = QtWidgets.QRadioButton("Trailing edge")
        self.radio_le.setChecked(True)
        edge_group = QtWidgets.QButtonGroup(self)
        edge_group.addButton(self.radio_le)
        edge_group.addButton(self.radio_te)
        self.radio_manual = QtWidgets.QRadioButton("Manual angle (swept back is positive)")
        self.spin_manual = FocusSpinBox()
        self.spin_manual.setRange(-89.0, 89.0)
        self.spin_manual.setDecimals(2)
        self.spin_manual.setSingleStep(1.0)
        self.spin_manual.setSuffix(" deg")
        self.spin_manual.setValue(MANUAL_ANGLE_DEFAULT_DEG)
        self.spin_manual.setLocale(QtCore.QLocale(QtCore.QLocale.Language.English))
        source = QtWidgets.QButtonGroup(self)
        source.addButton(self.radio_wing)
        source.addButton(self.radio_manual)
        has_wings = bool(self.wings)
        self.radio_wing.setEnabled(has_wings)
        self.combo_wing.setEnabled(has_wings)
        (self.radio_wing if has_wings else self.radio_manual).setChecked(True)
        self.info = QtWidgets.QLabel()
        self.info.setWordWrap(True)
        self.preview = QtWidgets.QLabel()
        self.preview.setWordWrap(True)
        self.buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok
                                                  | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

        form = QtWidgets.QVBoxLayout(self)
        form.addWidget(self.radio_wing)
        row = QtWidgets.QHBoxLayout()
        row.addSpacing(18)
        row.addWidget(self.combo_wing, 1)
        row.addWidget(self.radio_le)
        row.addWidget(self.radio_te)
        form.addLayout(row)
        form.addWidget(self.radio_manual)
        row = QtWidgets.QHBoxLayout()
        row.addSpacing(18)
        row.addWidget(self.spin_manual)
        row.addStretch(1)
        form.addLayout(row)
        form.addWidget(self.info)
        form.addWidget(self.preview)
        form.addWidget(self.buttons)
        for w in (self.radio_wing, self.radio_manual, self.radio_le, self.radio_te):
            w.toggled.connect(self.refresh)
        self.combo_wing.currentIndexChanged.connect(self.refresh)
        self.spin_manual.valueChanged.connect(self.refresh)
        self.refresh()

    def choice(self) -> tuple[float, str] | None:
        """(edge sweep in degrees, label), or None when no wing is available for a wing choice."""
        if self.radio_manual.isChecked() or not self.wings:
            return float(self.spin_manual.value()), "manual"
        wing = self.wings[self.combo_wing.currentIndex()]
        edge = "LE" if self.radio_le.isChecked() else "TE"
        return vspio.edge_sweep_at(wing, self.y, edge).angle_deg, f"{wing.name} {edge}"

    def blocked_reason(self) -> str | None:
        angle, _ = self.choice()
        factor = InletPlane((0.0, 0.0, 0.0), self.pitch_deg, sweep_for_edge_angle(angle)).projection_factor
        if factor < MIN_ALIGN_PROJECTION:
            return (f"Blocked: |n.X| would be {factor:.2f} (< {MIN_ALIGN_PROJECTION:g}): the opening would be more "
                    f"than {math.degrees(math.acos(MIN_ALIGN_PROJECTION)):.0f} deg off the flow")
        return None

    def refresh(self) -> None:
        wing_mode = self.radio_wing.isChecked() and bool(self.wings)
        self.combo_wing.setEnabled(wing_mode)
        self.radio_le.setEnabled(wing_mode)
        self.radio_te.setEnabled(wing_mode)
        self.spin_manual.setEnabled(not wing_mode)
        notes = []
        if wing_mode:
            wing = self.wings[self.combo_wing.currentIndex()]
            le, te = vspio.edge_sweep_at(wing, self.y, "LE"), vspio.edge_sweep_at(wing, self.y, "TE")
            notes.append(f"Section {le.section_index + 1} of {le.n_sections} (y {le.y0:.2f} .. {le.y1:.2f} m): "
                         f"LE Λ = {le.angle_deg:.1f}°, TE Λ = {te.angle_deg:.1f}°")
            if le.outside:
                notes.append("The inlet's y is outside the wing span: the nearest section is used.")
            if wing.rotated:
                notes.append("This wing is rotated: its sweeps are given in its own frame.")
        elif not self.wings:
            notes.append("There is no Wing geom in the model: enter an angle.")
        self.info.setText("\n".join(notes))
        angle, label = self.choice()
        reason = self.blocked_reason()
        sweep = sweep_for_edge_angle(angle)
        self.preview.setText(reason or f"{label}: Λ = {angle:.1f}° gives sweep {sweep:.2f}°")
        self.preview.setStyleSheet(f"color: {COLORS['error' if reason else 'info']};")
        self.buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Ok).setEnabled(reason is None)


class ExploreWindow(QtWidgets.QMainWindow):
    def __init__(self, scene: Scene, model: InletModel, with_3d: bool = True, parent=None):
        super().__init__(parent)
        self.scene, self.model = scene, model
        self.plotter = None
        self.view: SceneView | None = None
        self._syncing = False
        self._press_pos: tuple[int, int] | None = None
        self.setStyleSheet(STYLESHEET)
        self.resize(*WINDOW_SIZE)
        self._build_center(with_3d)
        self._build_actions()
        self._build_panel()
        self._build_sketch_dock()
        self._build_toolbar()
        self._build_status_bar()
        self.model.add_listener(self.sync)
        self.sync()

    # ---- construction ------------------------------------------------------------------------------------
    def _build_center(self, with_3d: bool) -> None:
        if not with_3d:
            self.setCentralWidget(QtWidgets.QLabel("(3D view disabled)", alignment=QtCore.Qt.AlignmentFlag.AlignCenter))
            return
        from pyvistaqt import QtInteractor

        self.plotter = QtInteractor(self)
        self.setCentralWidget(self.plotter)
        self.view = SceneView(self.plotter, self.scene)
        iren = self.plotter.iren
        iren.add_observer("LeftButtonPressEvent", self._on_left_press)
        iren.add_observer("LeftButtonReleaseEvent", self._on_left_release)

    def _action(self, text: str, slot, shortcuts=(), checkable=False, tip: str = "") -> QtGui.QAction:
        act = QtGui.QAction(text, self)
        act.setCheckable(checkable)
        if shortcuts:
            act.setShortcuts([QtGui.QKeySequence(s) for s in shortcuts])
        if tip:
            act.setToolTip(tip)
        act.triggered.connect(lambda *_: slot())
        self.addAction(act)  # makes the shortcut live for the whole window
        return act

    def _build_actions(self) -> None:
        m = self.model
        self.act_place = self._action("Place inlet", m.toggle_place, ["P"], True,
                                      "Place inlet (P): the next click on the fuselage places the plane origin there")
        self.act_cancel = self._action("Cancel placement", m.cancel_place, [QtCore.Qt.Key.Key_Escape.value])
        self.act_undo = self._action("Undo", m.undo, ["Ctrl+Z"], tip="Undo (Ctrl+Z)")
        self.act_redo = self._action("Redo", m.redo, ["Ctrl+Y", "Ctrl+Shift+Z"], tip="Redo (Ctrl+Y)")
        self.act_save = self._action("Save to config", m.save, ["Ctrl+S"], tip="Save to config (Ctrl+S)")
        self.act_revert = self._action("Revert to saved", m.revert, tip="Re-read the inlet from the config file")
        self.act_fit_area = self._action("Fit area", m.fit_area, tip="Scale the free vertices to the target area")
        self.act_align = self._action("Align to edge…", self._on_align,
                                      tip="Set the sweep so the opening's planform trace is parallel to a wing "
                                          "edge or to a given angle")
        self.view_actions = {
            name: self._action(name.capitalize(), lambda n=name: self._set_view(n), [str(i + 1)],
                               tip=f"{name.capitalize()} view ({i + 1})")
            for i, name in enumerate(("side", "top", "front", "iso"))
        }
        self.act_context = self._action("Context", lambda: self._toggle("context", self.act_context), checkable=True,
                                        tip="Show / hide the other components (wing, fins, stores...)")
        self.act_envelope = self._action("Envelope", lambda: self._toggle("envelope", self.act_envelope),
                                         checkable=True, tip="Show / hide the engine envelope outline")
        for act, name in ((self.act_context, "context"), (self.act_envelope, "envelope")):
            act.setChecked(True)
            act.setEnabled(self.view is not None and self.view.has(name))

    def _build_toolbar(self) -> None:
        bar = self.addToolBar("View")
        bar.setMovable(False)
        for act in (self.act_undo, self.act_redo):
            bar.addAction(act)
        bar.addSeparator()
        for act in self.view_actions.values():
            bar.addAction(act)
        bar.addSeparator()
        bar.addAction(self.act_context)
        bar.addAction(self.act_envelope)
        bar.addSeparator()
        self.act_sketch = self.sketch_dock.toggleViewAction()
        self.act_sketch.setText("Sketch")
        self.act_sketch.setToolTip("Show / hide the section sketcher")
        bar.addAction(self.act_sketch)

    def _spin(self, lo, hi, step, decimals=3, suffix="") -> FocusSpinBox:
        box = FocusSpinBox()
        box.setRange(lo, hi)
        box.setSingleStep(step)
        box.setDecimals(decimals)
        if suffix:
            box.setSuffix(suffix)
        box.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        box.setLocale(QtCore.QLocale(QtCore.QLocale.Language.English))  # '.' decimals, like the readouts
        return box

    def _build_panel(self) -> None:
        panel = QtWidgets.QWidget()
        col = QtWidgets.QVBoxLayout(panel)
        col.setContentsMargins(6, 4, 6, 6)
        col.setSpacing(4)

        # Placement
        self.grp_place = CollapsibleGroup("Placement")
        self.btn_place = QtWidgets.QPushButton("Place inlet  (P)")
        self.btn_place.setCheckable(True)
        self.btn_place.setToolTip("While on, the next click on the fuselage places the plane origin, then this "
                                  "switches off. Esc cancels.")
        self.btn_place.clicked.connect(lambda *_: self.model.toggle_place())
        self.chk_lock = QtWidgets.QCheckBox("Lock inlet")
        self.chk_lock.setToolTip("Disables placement and nudges")
        self.chk_lock.toggled.connect(self._on_lock)
        self.grp_place.form.addRow(self.btn_place)
        self.grp_place.form.addRow(self.chk_lock)
        col.addWidget(self.grp_place)

        # Inlet plane: origin, pitch, sweep
        b = self.scene.fuselage.bounds
        pad = POSITION_MARGIN * max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
        self.grp_position = CollapsibleGroup("Inlet plane")
        self.spin_pos = {
            "x": self._spin(b[0] - pad, b[1] + pad, DEFAULT_NUDGE_STEP, suffix=" m"),
            "y": self._spin(0.0, b[3] + pad, DEFAULT_NUDGE_STEP, suffix=" m"),
            "z": self._spin(b[4] - pad, b[5] + pad, DEFAULT_NUDGE_STEP, suffix=" m"),
        }
        tips = {"x": "Origin X (aft), m", "y": "Origin Y (right inlet, y >= 0), m", "z": "Origin Z, m"}
        for axis, box in self.spin_pos.items():
            box.setToolTip(tips[axis])
            box.valueChanged.connect(lambda *_: self._on_position())
            self.grp_position.form.addRow(f"Origin {axis.upper()}", box)
        self.combo_step = QtWidgets.QComboBox()
        for step in NUDGE_STEPS:
            self.combo_step.addItem(f"{step:g} m", step)
        self.combo_step.setCurrentIndex(NUDGE_STEPS.index(DEFAULT_NUDGE_STEP))
        self.combo_step.setToolTip("Step of the arrow keys / spin arrows (nudge)")
        self.combo_step.currentIndexChanged.connect(self._on_step)
        self.grp_position.form.addRow("Nudge step", self.combo_step)
        self.spin_pitch = self._spin(*PITCH_RANGE_DEG, 1.0, decimals=2, suffix=" deg")
        self.spin_pitch.setToolTip("Rotation about Y: positive tilts the opening up")
        self.spin_sweep = self._spin(*SWEEP_RANGE_DEG, 1.0, decimals=2, suffix=" deg")
        self.spin_sweep.setToolTip("Rotation about Z: the opening swept in planform")
        self.spin_pitch.valueChanged.connect(lambda v: self._on_param("pitch_deg", v))
        self.spin_sweep.valueChanged.connect(lambda v: self._on_param("sweep_deg", v))
        self.grp_position.form.addRow("Pitch", self.spin_pitch)
        self.grp_position.form.addRow("Sweep", self.spin_sweep)
        self.btn_align = QtWidgets.QPushButton("Align to edge…")
        self.btn_align.setToolTip(self.act_align.toolTip())
        self.btn_align.clicked.connect(lambda *_: self._on_align())
        self.grp_position.form.addRow(self.btn_align)
        col.addWidget(self.grp_position)

        # Readouts
        self.grp_readouts = CollapsibleGroup("Readouts")
        self.readout_view = QtWidgets.QTextEdit()
        self.readout_view.setReadOnly(True)
        self.readout_view.setFont(QtGui.QFont(READOUT_FONT, READOUT_FONT_POINT_SIZE))
        self.readout_view.setMinimumHeight(READOUT_MIN_HEIGHT)
        self.readout_view.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.WidgetWidth)
        self.grp_readouts.form.addRow(self.readout_view)
        col.addWidget(self.grp_readouts)

        # Save / revert
        row = QtWidgets.QHBoxLayout()
        self.btn_save = QtWidgets.QPushButton("Save to config  (Ctrl+S)")
        self.btn_revert = QtWidgets.QPushButton("Revert to saved")
        self.btn_save.clicked.connect(lambda *_: self.model.save())
        self.btn_revert.clicked.connect(lambda *_: self.model.revert())
        row.addWidget(self.btn_save)
        row.addWidget(self.btn_revert)
        col.addLayout(row)
        col.addStretch(1)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(panel)
        self.dock = QtWidgets.QDockWidget("Inlet", self)
        self.dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.dock.setTitleBarWidget(QtWidgets.QWidget())  # no title bar: the panel is part of the window
        self.dock.setWidget(scroll)
        self.dock.setFixedWidth(PANEL_WIDTH)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.dock)

    def _build_sketch_dock(self) -> None:
        self.sketch = SketchDock(self.model)
        self.sketch_dock = QtWidgets.QDockWidget("Section sketch", self)
        self.sketch_dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
                                     | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
                                     | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.sketch_dock.setWidget(self.sketch)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.sketch_dock)
        self.sketch_dock.setMinimumWidth(SKETCH_WIDTH)
        self.resizeDocks([self.sketch_dock], [SKETCH_WIDTH], QtCore.Qt.Orientation.Horizontal)

    def _build_status_bar(self) -> None:
        self.mode_label = QtWidgets.QLabel()
        self.message_label = QtWidgets.QLabel()
        self.statusBar().addWidget(self.mode_label, 1)
        self.statusBar().addPermanentWidget(self.message_label)

    # ---- widget -> model ------------------------------------------------------------------------------------
    def _on_param(self, key: str, value) -> None:
        if not self._syncing:
            self.model.set_param(key, value)

    def _on_position(self) -> None:
        if not self._syncing:
            self.model.set_origin(*(self.spin_pos[a].value() for a in "xyz"))

    def _on_lock(self, locked: bool) -> None:
        if not self._syncing:
            self.model.set_locked(locked)

    def ask_edge(self) -> tuple[float, str] | None:
        """Show the edge picker; (edge sweep in degrees, label) or None when cancelled."""
        plane = self.model.state.plane
        dialog = AlignEdgeDialog(self.scene.wings, plane.origin[1], plane.pitch_deg, self)
        return dialog.choice() if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted else None

    def _on_align(self) -> None:
        if not self.model.placed:
            return
        choice = self.ask_edge()
        if choice is not None:
            self.model.align_to_edge(*choice)

    def _on_step(self) -> None:
        step = self.combo_step.currentData()
        for box in self.spin_pos.values():
            box.setSingleStep(step)

    def _set_view(self, name: str) -> None:
        if self.view is not None:
            self.view.set_view(name)
            self.plotter.render()

    def _toggle(self, name: str, act: QtGui.QAction) -> None:
        if self.view is not None:
            self.view.set_visible(name, act.isChecked())
            self.plotter.render()

    # ---- clicks in the 3D view -------------------------------------------------------------------------------
    def _event_position(self) -> tuple[int, int]:
        x, y = self.plotter.iren.interactor.GetEventPosition()
        return int(x), int(y)

    def _on_left_press(self, *_) -> None:
        self._press_pos = self._event_position()

    def _on_left_release(self, *_) -> None:
        if self.model.mode != MODE_PLACING or self._press_pos is None:
            return  # outside place mode a click does nothing
        x, y = self._event_position()
        moved = np.hypot(x - self._press_pos[0], y - self._press_pos[1])
        self._press_pos = None
        if moved > CLICK_TOLERANCE_PX:
            return  # a drag rotates the camera
        point = self.view.pick_fuselage(x, y)
        if point is None:
            self.message_label.setText("No fuselage under the cursor")
            return
        self.model.handle_click(point)

    # ---- model -> widgets ---------------------------------------------------------------------------------------
    def _set_value(self, box: QtWidgets.QDoubleSpinBox, value: float) -> None:
        box.blockSignals(True)
        box.setValue(value)
        box.blockSignals(False)

    def sync(self) -> None:
        """Bring every widget, the title, the status bar and the 3D view in line with the model."""
        m = self.model
        self._syncing = True
        try:
            placed = m.placed
            if placed:
                for axis, value in zip("xyz", m.state.plane.origin):
                    self._set_value(self.spin_pos[axis], float(value))
                self._set_value(self.spin_pitch, m.state.plane.pitch_deg)
                self._set_value(self.spin_sweep, m.state.plane.sweep_deg)
            self.chk_lock.setChecked(m.locked)
            placing = m.mode == MODE_PLACING
            for checkable in (self.btn_place, self.act_place):
                checkable.setChecked(placing)
            for box in self.spin_pos.values():
                box.setEnabled(placed and not m.locked)
            self.combo_step.setEnabled(placed and not m.locked)
            for box in (self.spin_pitch, self.spin_sweep):
                box.setEnabled(placed)
            self.btn_align.setEnabled(placed)
            self.btn_place.setEnabled(not m.locked)
            self.act_undo.setEnabled(m.can_undo)
            self.act_redo.setEnabled(m.can_redo)
            can_save = m.can_save
            self.btn_save.setEnabled(can_save)
            self.act_save.setEnabled(can_save)
            tip = ("Save to config (Ctrl+S)" if can_save else
                   "Save is disabled: the example config is read-only" if m.save_target is None else
                   "Save is disabled: the opening is not valid" if m.placed else "Place an inlet first")
            self.btn_save.setToolTip(tip)
            self.readout_view.setHtml(readouts_html(m.readouts()))
        finally:
            self._syncing = False
        self.mode_label.setText(m.mode_text)
        self.message_label.setText(m.message)
        self.setWindowTitle(self.window_title())
        if self.plotter is not None:
            self.view.update_inlet(m)
            self.plotter.setCursor(QtCore.Qt.CursorShape.CrossCursor if m.mode == MODE_PLACING
                                   else QtCore.Qt.CursorShape.ArrowCursor)
            self.plotter.render()

    def window_title(self) -> str:
        return f"VSP Duct Manager - inlet explorer - {self.model.config_path.name}" + (" *" if self.model.dirty else "")

    # ---- closing ----------------------------------------------------------------------------------------------------
    def ask_unsaved(self) -> str:
        """Ask what to do with unsaved changes: 'save', 'discard' or 'cancel'."""
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle("Unsaved changes")
        box.setText("The inlet has unsaved changes.")
        can_save = self.model.can_save
        if can_save:
            save = box.addButton("Save", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        else:
            box.setInformativeText("The example config is read-only, so the changes cannot be saved."
                                   if self.model.save_target is None else
                                   "The opening is not valid, so the changes cannot be saved.")
            save = None
        discard = box.addButton("Discard", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
        cancel = box.addButton("Cancel", QtWidgets.QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(save or cancel)
        box.exec()
        clicked = box.clickedButton()
        return "save" if clicked is save and save is not None else "discard" if clicked is discard else "cancel"

    def closeEvent(self, event) -> None:
        if self.model.dirty:
            choice = self.ask_unsaved()
            if choice == "cancel" or (choice == "save" and not self.model.save()):
                event.ignore()
                return
        if self.plotter is not None:
            self.plotter.close()
        event.accept()


def run(scene: Scene, model: InletModel) -> int:
    """Open the explorer window and run the Qt event loop."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ExploreWindow(scene, model)
    window.show()
    return app.exec()
