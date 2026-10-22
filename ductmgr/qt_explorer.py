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

from ductmgr import exportrun, vspio
from ductmgr.explore_model import MIN_ALIGN_PROJECTION, MODE_PLACING, InletModel, Readout
from ductmgr.geometry import InletPlane, sweep_for_edge_angle
from ductmgr.qt_paths import AreaCanvas, PathCanvas
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
SECTION_LIST_HEIGHT = 110
INLET_LIST_HEIGHT = 96
BODY_TABLE_HEIGHT = 150
PATHS_HEIGHT = 300  # the 2D duct tabs dock at the bottom
HEAVY_DELAY_MS = 150  # bodies and checks are rebuilt this long after the last change (the rest updates at once)
WALL_RANGE_M = (0.0005, 0.2)
LIP_RANGE_M = (0.0, 0.5)
NEW_SECTION_S = 0.5  # the s suggested for a new key section
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
        self.export_runner = exportrun.export_design  # replaced in tests
        self.last_export_text = ""
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
        self._build_paths_dock()
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
        self.act_export = self._action("Export to VSP", self._on_export, ["Ctrl+E"],
                                       tip="Write the duct into <model>_ducts.vsp3 and verify it (Ctrl+E)")
        self.act_fit_area = self._action("Fit area", m.fit_area, tip="Scale the free vertices to the target area")
        self.act_align = self._action("Align to edge…", self._on_align,
                                      tip="Set the sweep so the opening's planform trace is parallel to a wing "
                                          "edge or to a given angle")
        self.view_actions = {
            name: self._action(name.capitalize(), lambda n=name: self._set_view(n), [str(i + 1)],
                               tip=f"{name.capitalize()} view ({i + 1})")
            for i, name in enumerate(("side", "top", "front", "iso"))
        }
        self.act_context = self._action("Context", lambda: self.model.set_context_visible(self.act_context.isChecked()),
                                        checkable=True,
                                        tip="Show / hide every body that is not a skin body or the engine, in all views")
        self.act_envelope = self._action("Envelope", lambda: self._toggle("envelope", self.act_envelope),
                                         checkable=True, tip="Show / hide the engine envelope outline")
        self.act_context.setChecked(True)
        self.act_envelope.setChecked(True)
        self.act_envelope.setEnabled(self.view is not None and self.view.has("envelope"))

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
        self.act_paths = self.paths_dock.toggleViewAction()
        self.act_paths.setText("Duct tabs")
        self.act_paths.setToolTip("Show / hide the side / top / area law editors")
        bar.addAction(self.act_paths)

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

        # Inlets: the list, add / duplicate / remove / rename, mirror, flow fraction, global symmetry
        self.grp_inlets = CollapsibleGroup("Inlets")
        self.inlet_list = QtWidgets.QListWidget()
        self.inlet_list.setMaximumHeight(INLET_LIST_HEIGHT)
        self.inlet_list.setToolTip("The inlet the section editor, the 2D views and the panel edit. Mirrored twins are "
                                   "generated and not listed. The Trunk row (always last, fixed) selects the trunk: its "
                                   "key sections in the section editor and its area curve in the Area tab.")
        self.inlet_list.currentRowChanged.connect(self._on_inlet_row)
        self.btn_add_inlet = QtWidgets.QPushButton("Add")
        self.btn_dup_inlet = QtWidgets.QPushButton("Duplicate")
        self.btn_del_inlet = QtWidgets.QPushButton("Remove")
        self.btn_ren_inlet = QtWidgets.QPushButton("Rename…")
        self.btn_add_inlet.setToolTip("Add an inlet, then click on the fuselage to place it")
        self.btn_dup_inlet.setToolTip("Copy the selected inlet (same plane and sections); then move the copy")
        self.btn_up_inlet = QtWidgets.QPushButton("Up")
        self.btn_down_inlet = QtWidgets.QPushButton("Down")
        self.btn_up_inlet.setToolTip("Move the selected inlet up the list (the trunk is not affected)")
        self.btn_down_inlet.setToolTip("Move the selected inlet down the list (the trunk is not affected)")
        self.btn_up_inlet.clicked.connect(lambda *_: self.model.move_inlet(self.model.selected, -1))
        self.btn_down_inlet.clicked.connect(lambda *_: self.model.move_inlet(self.model.selected, 1))
        self.btn_add_inlet.clicked.connect(lambda *_: self.model.add_inlet())
        self.btn_dup_inlet.clicked.connect(lambda *_: self.model.duplicate_inlet())
        self.btn_del_inlet.clicked.connect(lambda *_: self.model.remove_inlet())
        self.btn_ren_inlet.clicked.connect(lambda *_: self._on_rename_inlet())
        row1, row2, row3 = QtWidgets.QHBoxLayout(), QtWidgets.QHBoxLayout(), QtWidgets.QHBoxLayout()
        for b in (self.btn_add_inlet, self.btn_dup_inlet):
            row1.addWidget(b)
        for b in (self.btn_del_inlet, self.btn_ren_inlet):
            row2.addWidget(b)
        for b in (self.btn_up_inlet, self.btn_down_inlet):
            row3.addWidget(b)
        self.chk_mirror = QtWidgets.QCheckBox("Mirrored twin")
        self.chk_mirror.setToolTip("With symmetry xz: also generate the mirror image of this inlet about y = 0 "
                                   "(not for an inlet centred on y = 0, which is its own mirror image)")
        self.chk_mirror.toggled.connect(self._on_mirror)
        self.spin_flow = self._spin(0.1, 100.0, 1.0, decimals=2, suffix=" %")
        self.spin_flow.setToolTip("Share of the total flow carried by this branch (each physical branch); all shares "
                                  "are normalised to sum to 100 %")
        self.spin_flow.valueChanged.connect(self._on_flow)
        self.btn_flow_equal = QtWidgets.QPushButton("Equal")
        self.btn_flow_equal.setToolTip("Back to an equal share")
        self.btn_flow_equal.clicked.connect(lambda *_: self.model.set_flow_fraction(self.model.selected, None))
        flow_row = QtWidgets.QHBoxLayout()
        flow_row.addWidget(self.spin_flow, 1)
        flow_row.addWidget(self.btn_flow_equal)
        self.combo_symmetry = QtWidgets.QComboBox()
        self.combo_symmetry.addItem("xz (mirror about y = 0)", "xz")
        self.combo_symmetry.addItem("none (free layout)", "none")
        self.combo_symmetry.setToolTip("xz: mirrored twins, junction and trunk held on y = 0. none: every inlet is "
                                       "independent, no constraint on the junction")
        self.combo_symmetry.currentIndexChanged.connect(self._on_symmetry)
        self.grp_inlets.form.addRow(self.inlet_list)
        self.grp_inlets.form.addRow(row1)
        self.grp_inlets.form.addRow(row2)
        self.grp_inlets.form.addRow(row3)
        self.grp_inlets.form.addRow(self.chk_mirror)
        self.grp_inlets.form.addRow("Flow share", flow_row)
        self.grp_inlets.form.addRow("Symmetry", self.combo_symmetry)
        col.addWidget(self.grp_inlets)

        # Bodies: visibility and skin of every Geom
        self.grp_bodies = CollapsibleGroup("Bodies", expanded=False)
        self.body_table = QtWidgets.QTableWidget(0, 3)
        self.body_table.setHorizontalHeaderLabels(["Geom", "Show", "Skin"])
        self.body_table.verticalHeader().setVisible(False)
        self.body_table.setMaximumHeight(BODY_TABLE_HEIGHT)
        self.body_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.body_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        head = self.body_table.horizontalHeader()
        head.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for col_index in (1, 2):
            head.setSectionResizeMode(col_index, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.body_table.setToolTip("Show: hide a body in the 3D, Side and Top views (hidden bodies stay in the clash "
                                   "checks). Skin: the Geoms (their union) that form the fuselage skin for the "
                                   "section sketch, snapping, clearance and external-region checks.")
        self.body_table.itemChanged.connect(self._on_body_item)
        self.grp_bodies.form.addRow(self.body_table)
        col.addWidget(self.grp_bodies)

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
        b = self.model.scene_bounds()
        pad = POSITION_MARGIN * max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
        self.grp_position = CollapsibleGroup("Inlet plane")
        self.spin_pos = {
            "x": self._spin(b[0] - pad, b[1] + pad, DEFAULT_NUDGE_STEP, suffix=" m"),
            "y": self._spin(-(b[3] + pad), b[3] + pad, DEFAULT_NUDGE_STEP, suffix=" m"),
            "z": self._spin(b[4] - pad, b[5] + pad, DEFAULT_NUDGE_STEP, suffix=" m"),
        }
        tips = {"x": "Origin X (aft), m", "y": "Origin Y, m (a mirrored inlet is kept on the right, y >= 0)", "z": "Origin Z, m"}
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

        # Duct walls
        self.grp_duct = CollapsibleGroup("Duct walls")
        self.spin_wall = self._spin(*WALL_RANGE_M, 0.001, decimals=4, suffix=" m")
        self.spin_wall.setToolTip("Thickness of the positive body around the inner section (ducts.wall_thickness)")
        self.spin_lip = self._spin(*LIP_RANGE_M, 0.005, decimals=3, suffix=" m")
        self.spin_lip.setToolTip("How far the negative starts upstream of the opening plane (ducts.lip_extension)")
        self.spin_wall.valueChanged.connect(lambda v: self._on_duct("wall", v))
        self.spin_lip.valueChanged.connect(lambda v: self._on_duct("lip", v))
        self.lbl_lip_mode = QtWidgets.QLabel()
        self.btn_reset_cl = QtWidgets.QPushButton("Reset centreline")
        self.btn_reset_area = QtWidgets.QPushButton("Reset area law")
        self.btn_reset_cl.clicked.connect(lambda *_: self.model.reset_centreline())
        self.btn_reset_area.clicked.connect(lambda *_: self.model.reset_area())
        self.grp_duct.form.addRow("Wall thickness", self.spin_wall)
        self.grp_duct.form.addRow("Lip extension", self.spin_lip)
        self.grp_duct.form.addRow("Lip mode", self.lbl_lip_mode)
        resets = QtWidgets.QHBoxLayout()
        resets.addWidget(self.btn_reset_cl)
        resets.addWidget(self.btn_reset_area)
        self.grp_duct.form.addRow(resets)
        col.addWidget(self.grp_duct)

        # Key sections along the branch
        self.grp_sections = CollapsibleGroup("Key sections")
        self.section_list = QtWidgets.QListWidget()
        self.section_list.setMaximumHeight(SECTION_LIST_HEIGHT)
        self.section_list.setToolTip("The section shown in the editor. s = 0 is the inlet, s = 1 the fixed engine "
                                     "face; the others take their size from the area law A(s)")
        self.section_list.currentRowChanged.connect(self._on_section_row)
        self.btn_add_section = QtWidgets.QPushButton("Add…")
        self.btn_add_section.setToolTip("Add a key section at a chosen s (e.g. at the junction)")
        self.btn_remove_section = QtWidgets.QPushButton("Remove")
        self.btn_remove_section.setToolTip("Remove the selected intermediate key section")
        self.btn_add_section.clicked.connect(lambda *_: self._on_add_section())
        self.btn_remove_section.clicked.connect(lambda *_: self._on_remove_section())
        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.btn_add_section)
        buttons.addWidget(self.btn_remove_section)
        self.grp_sections.form.addRow(self.section_list)
        self.grp_sections.form.addRow(buttons)
        col.addWidget(self.grp_sections)

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
        self.btn_export = QtWidgets.QPushButton("Export to VSP  (Ctrl+E)")
        self.btn_export.clicked.connect(lambda *_: self._on_export())
        col.addWidget(self.btn_export)
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

    def _build_paths_dock(self) -> None:
        self.tabs = QtWidgets.QTabWidget()
        self.canvas_side = PathCanvas(self.model, "side")
        self.canvas_top = PathCanvas(self.model, "top")
        self.canvas_area = AreaCanvas(self.model)
        self.tabs.addTab(self.canvas_side, "Side (x-z)")
        self.tabs.addTab(self.canvas_top, "Top (x-y)")
        self.tabs.addTab(self.canvas_area, "Area law A(s)")
        self.paths_dock = QtWidgets.QDockWidget("Duct", self)
        self.paths_dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
                                    | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
                                    | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetClosable)
        self.paths_dock.setWidget(self.tabs)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.paths_dock)
        self.paths_dock.setMinimumHeight(PATHS_HEIGHT)
        self.resizeDocks([self.paths_dock], [PATHS_HEIGHT], QtCore.Qt.Orientation.Vertical)
        self._heavy_timer = QtCore.QTimer(self)
        self._heavy_timer.setSingleShot(True)
        self._heavy_timer.setInterval(HEAVY_DELAY_MS)
        self._heavy_timer.timeout.connect(self.refresh_heavy)
        self.model.live_checks = True

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

    def _on_duct(self, which: str, value: float) -> None:
        if self._syncing:
            return
        if which == "wall":
            self.model.set_wall_thickness(value)
        else:
            self.model.set_lip_extension(value)

    def _on_body_item(self, item) -> None:
        """A Show / Skin checkbox of the Bodies list."""
        if self._syncing:
            return
        name = self.body_table.item(item.row(), 0).text()
        checked = item.checkState() == QtCore.Qt.CheckState.Checked
        if item.column() == 1:
            self.model.set_body_visible(name, checked)
        elif item.column() == 2:
            if not self.model.toggle_skin_body(name, checked):
                self.sync()  # refused (the last skin body) or unchanged: put the box back

    def _sync_bodies(self, m) -> None:
        rows = m.body_rows()
        self.body_table.blockSignals(True)
        self.body_table.setRowCount(len(rows))
        for r, (name, visible, skin, kind) in enumerate(rows):
            name_item = QtWidgets.QTableWidgetItem(name)
            name_item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled)
            self.body_table.setItem(r, 0, name_item)
            for c, flag, enabled in ((1, visible, True), (2, skin, kind != "engine")):
                box = QtWidgets.QTableWidgetItem()
                flags = QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEnabled
                box.setFlags(flags if enabled else QtCore.Qt.ItemFlag.NoItemFlags)
                box.setCheckState(QtCore.Qt.CheckState.Checked if flag else QtCore.Qt.CheckState.Unchecked)
                self.body_table.setItem(r, c, box)
        self.body_table.blockSignals(False)
        self.act_context.setChecked(m.context_visible)
        self.act_context.setEnabled(any(kind == "context" for _, _, _, kind in rows))

    def _on_inlet_row(self, row: int) -> None:
        if self._syncing or row < 0:
            return
        if row == len(self.model.state.inlets):  # the fixed Trunk row
            self.model.select_trunk()
        else:
            self.model.select_inlet(row)

    def ask_inlet_name(self, current: str) -> str | None:
        """Ask for a new name for the selected inlet; None when cancelled."""
        text, ok = QtWidgets.QInputDialog.getText(self, "Rename inlet", "Name:", text=current)
        return text if ok else None

    def _on_rename_inlet(self) -> None:
        name = self.ask_inlet_name(self.model.inlet.name)
        if name is not None:
            self.model.rename_inlet(self.model.selected, name)

    def _on_mirror(self, flag: bool) -> None:
        if not self._syncing:
            self.model.set_mirror(self.model.selected, flag)

    def _on_flow(self, value: float) -> None:
        if not self._syncing:
            self.model.set_flow_fraction(self.model.selected, value / 100.0)

    def _on_symmetry(self, *_) -> None:
        if not self._syncing:
            self.model.set_symmetry(self.combo_symmetry.currentData())

    def _on_lock(self, locked: bool) -> None:
        if not self._syncing:
            self.model.set_locked(locked)

    def ask_section_s(self) -> float | None:
        """Ask for the s of a new key section; None when cancelled."""
        what = "s (0 = trunk start, 1 = engine face):" if self.model.trunk_selected else "s (0 = inlet, 1 = engine face):"
        s, ok = QtWidgets.QInputDialog.getDouble(self, "Add key section", what, NEW_SECTION_S, 0.0, 1.0, 3)
        return s if ok else None

    def _on_add_section(self) -> None:
        if not self.model.placed:
            return
        s = self.ask_section_s()
        if s is not None:
            self.model.add_key_section(s)

    def _on_remove_section(self) -> None:
        self.model.remove_key_section(self.model.active)

    def _on_section_row(self, row: int) -> None:
        if not self._syncing and row >= 0:
            self.model.select_section(row)

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
            inlet_edit = placed and not m.trunk_selected  # the plane belongs to the inlet, not to the trunk
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
                box.setEnabled(inlet_edit and not m.locked)
            self.combo_step.setEnabled(inlet_edit and not m.locked)
            for box in (self.spin_pitch, self.spin_sweep):
                box.setEnabled(inlet_edit)
            self.btn_align.setEnabled(inlet_edit)
            self._sync_inlets(m)
            self._sync_bodies(m)
            self._set_value(self.spin_wall, m.state.duct.wall_thickness)
            self._set_value(self.spin_lip, m.state.duct.lip_extension)
            self.lbl_lip_mode.setText(m.state.duct.lip_mode)
            for box in (self.spin_wall, self.spin_lip, self.btn_reset_cl, self.btn_reset_area):
                box.setEnabled(placed)
            self.section_list.clear()
            for s, label, _ in m.section_list():
                self.section_list.addItem(f"s = {s:.3f}   {label}")
            if placed:
                self.section_list.setCurrentRow(m.active)
            self.btn_add_section.setEnabled(placed)
            self.btn_remove_section.setEnabled(placed and 0 < m.active < m.n_sections)
            self.btn_place.setEnabled(not m.locked and not m.trunk_selected)
            self.act_undo.setEnabled(m.can_undo)
            self.act_redo.setEnabled(m.can_redo)
            can_save = m.can_save
            self.btn_save.setEnabled(can_save)
            self.act_save.setEnabled(can_save)
            tip = ("Save to config (Ctrl+S)" if can_save else
                   "Save is disabled: the example config is read-only" if m.save_target is None else
                   "Save is disabled: the opening is not valid" if m.placed else "Place an inlet first")
            self.btn_save.setToolTip(tip)
            readouts = m.readouts()
            self.readout_view.setHtml(readouts_html(readouts))
            blocker = m.export_blocker(readouts)
            self.btn_export.setEnabled(blocker is None)
            self.act_export.setEnabled(blocker is None)
            self.btn_export.setToolTip("Write the duct into <model>_ducts.vsp3 and verify it (Ctrl+E)" if blocker is None
                                       else f"Export is disabled: {blocker}")
        finally:
            self._syncing = False
        self.mode_label.setText(m.mode_text)
        self.message_label.setText(m.message)
        self.setWindowTitle(self.window_title())
        for canvas in (self.canvas_side, self.canvas_top, self.canvas_area):
            canvas.update()
        if m.placed and m.can_build:
            self._heavy_timer.start()
        if self.plotter is not None:
            self.view.update_bodies(m)
            self.view.update_inlet(m)
            self.view.update_ducts(m)
            self.plotter.setCursor(QtCore.Qt.CursorShape.CrossCursor if m.mode == MODE_PLACING
                                   else QtCore.Qt.CursorShape.ArrowCursor)
            self.plotter.render()

    def _sync_inlets(self, m) -> None:
        """The inlets list and its controls."""
        st = m.state
        fractions, _ = m._fractions(st)
        self.inlet_list.clear()
        for k, ins in enumerate(st.inlets):
            if not ins.placed:
                status = "not placed"
            else:
                tag = "self-symmetric" if ins.self_symmetric else "+ mirror" if m.has_twin(k) else ""
                status = f"{100 * fractions[k]:.1f} %" + (f", {tag}" if tag else "")
            self.inlet_list.addItem(f"{ins.name}   ({status})")
        trunk_item = QtWidgets.QListWidgetItem(f"Trunk   ({len(st.trunk)} section(s))")
        font = trunk_item.font()
        font.setBold(True)
        trunk_item.setFont(font)
        self.inlet_list.addItem(trunk_item)  # always last; cannot be removed, renamed or moved
        self.inlet_list.setCurrentRow(len(st.inlets) if m.trunk_selected else m.selected)
        ins = m.inlet
        on_inlet = not m.trunk_selected  # the inlet's own controls do not apply to the trunk
        self.btn_del_inlet.setEnabled(on_inlet and len(st.inlets) > 1)
        self.btn_dup_inlet.setEnabled(on_inlet and ins.placed)
        self.btn_ren_inlet.setEnabled(on_inlet)
        self.btn_up_inlet.setEnabled(on_inlet and m.selected > 0)
        self.btn_down_inlet.setEnabled(on_inlet and m.selected < len(st.inlets) - 1)
        self.chk_mirror.setChecked(ins.mirror)
        self.chk_mirror.setEnabled(on_inlet and st.symmetry == "xz" and not ins.self_symmetric)
        f = fractions.get(m.selected)
        self.spin_flow.setEnabled(on_inlet and f is not None)
        if f is not None:
            self._set_value(self.spin_flow, 100.0 * f)
        self.btn_flow_equal.setEnabled(on_inlet and f is not None and ins.flow_fraction is not None)
        self.combo_symmetry.setCurrentIndex(self.combo_symmetry.findData(st.symmetry))

    def refresh_heavy(self) -> None:
        """Build the bodies and run the checks for the current state, then redraw (called after a short pause
        in the edits, so dragging stays responsive)."""
        try:
            built = self.model.ensure_built()
        except Exception as exc:  # a failing check must not take the window down
            self.message_label.setText(f"Duct checks failed: {type(exc).__name__}: {exc}"[:200])
            return
        if not built:
            return
        self._syncing = True
        try:
            self.readout_view.setHtml(readouts_html(self.model.readouts()))
        finally:
            self._syncing = False
        if self.plotter is not None:
            self.view.update_ducts(self.model)
            self.plotter.render()

    # ---- export to OpenVSP ---------------------------------------------------------------------------------------------
    def _on_export(self) -> None:
        """Export the current design (Ctrl+E): disabled while an ERROR shows. Runs in this thread with a progress dialog;
        the result is printed to the terminal and shown in a dialog."""
        m = self.model
        blocker = m.export_blocker()
        if blocker is not None:
            self.message_label.setText(f"Export disabled: {blocker}"[:200])
            return
        progress = QtWidgets.QProgressDialog("Exporting to OpenVSP...", None, 0, 0, self)
        progress.setWindowTitle("Export to VSP")
        progress.setMinimumDuration(0)
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.show()

        def step(text: str) -> None:
            progress.setLabelText(text + "...")
            print(f"... {text}", flush=True)
            QtWidgets.QApplication.processEvents()

        try:
            inputs = m.export_inputs()
            result = self.export_runner(**inputs, progress=step)
        except Exception as exc:  # the dialog reports it; the window stays up
            progress.close()
            self.show_export_result(False, f"Export failed: {type(exc).__name__}: {exc}")
            return
        progress.close()
        print(result.text, flush=True)
        self.show_export_result(result.ok, result.text, result.warned)
        m._say(f"Exported to {result.out_path.name}: " + ("FAIL" if not result.ok else "PASS with warnings"
                                                             if result.warned else "PASS"))
        m._notify()

    def show_export_result(self, ok: bool, text: str, warned: bool = False) -> None:
        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Icon.Information if ok and not warned else QtWidgets.QMessageBox.Icon.Warning)
        box.setWindowTitle("Export to VSP")
        box.setText("Export finished: " + ("PASS" if ok and not warned else "PASS with warnings" if ok else "FAIL"))
        box.setDetailedText(text)
        box.setInformativeText(text.splitlines()[0] if text else "")
        self.last_export_text = text
        box.exec()

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
