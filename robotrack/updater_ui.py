"""The Update button's dialog, and the threads behind it.

Kept out of gui.py because updating is a self-contained interaction with its own
failure modes, and because everything here is Qt glue over ``robotrack.update``
-- the logic that decides what to download and how to apply it lives there and is
testable without a display.

The interaction is deliberately three explicit steps rather than one silent one:
check, review what changed, then install. An analysis tool that swaps its own
measurement code out from under a half-finished experiment is not being helpful,
so nothing is downloaded until the version and its notes have been seen.
"""

from __future__ import annotations

import traceback
from pathlib import Path

from PySide6.QtCore import QThread, Qt, QTimer, Signal
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QProgressBar,
                               QPushButton, QTextEdit, QVBoxLayout, QWidget)

from . import update as U


class CheckWorker(QThread):
    """Network or disk access, off the UI thread."""
    done = Signal(object, str)          # Release | None, error

    def __init__(self, channel: str):
        super().__init__()
        self.channel = channel

    def run(self):
        try:
            self.done.emit(U.check(self.channel), "")
        except U.UpdateError as exc:
            self.done.emit(None, str(exc))
        except Exception:
            self.done.emit(None, traceback.format_exc())


class DownloadWorker(QThread):
    progress = Signal(int, int)
    done = Signal(object, str)          # Path | None, error

    def __init__(self, rel: U.Release):
        super().__init__()
        self.rel = rel

    def run(self):
        try:
            path = U.download(self.rel, progress=lambda a, b: self.progress.emit(a, b))
            self.done.emit(path, "")
        except U.UpdateError as exc:
            self.done.emit(None, str(exc))
        except Exception:
            self.done.emit(None, traceback.format_exc())


class UpdateDialog(QDialog):
    """Check → review → install → relaunch."""

    relaunchRequested = Signal()

    def __init__(self, channel: str, parent=None, on_channel_change=None):
        super().__init__(parent)
        from . import APP_NAME
        self.setWindowTitle(f"{APP_NAME} — updates")
        self.setMinimumWidth(560)
        self._channel = channel
        self._on_channel_change = on_channel_change
        self._rel: U.Release | None = None
        self._worker = None
        # Set once an update has actually been written to disk. The caller acts
        # on it after exec() returns; see _downloaded.
        self.installed_version = ""

        v = QVBoxLayout(self)
        v.setContentsMargins(18, 16, 18, 16)
        v.setSpacing(11)

        self.lbl_version = QLabel(f"Installed version <b>{U.current_version()}</b>")
        v.addWidget(self.lbl_version)

        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        h.addWidget(QLabel("Channel"))
        self.edit_channel = QLineEdit(channel)
        self.edit_channel.setPlaceholderText(
            r"folder, e.g. C:\Users\you\Nextcloud2\robotrack-updates   ·   or github:owner/repo")
        h.addWidget(self.edit_channel, 1)
        v.addWidget(row)

        self.lbl_status = QLabel("Press Check for updates.")
        self.lbl_status.setWordWrap(True)
        v.addWidget(self.lbl_status)

        self.notes = QTextEdit()
        self.notes.setReadOnly(True)
        self.notes.setMinimumHeight(130)
        self.notes.setVisible(False)
        v.addWidget(self.notes)

        self.bar = QProgressBar()
        self.bar.setVisible(False)
        v.addWidget(self.bar)

        self.buttons = QDialogButtonBox()
        self.btn_check = QPushButton("Check for updates")
        self.btn_install = QPushButton("Download and install")
        self.btn_install.setProperty("primary", True)
        self.btn_install.setEnabled(False)
        self.btn_close = QPushButton("Close")
        self.buttons.addButton(self.btn_check, QDialogButtonBox.ActionRole)
        self.buttons.addButton(self.btn_install, QDialogButtonBox.AcceptRole)
        self.buttons.addButton(self.btn_close, QDialogButtonBox.RejectRole)
        v.addWidget(self.buttons)

        self.btn_check.clicked.connect(self.check)
        self.btn_install.clicked.connect(self.install)
        self.btn_close.clicked.connect(self.reject)

    # ---- channel ---------------------------------------------------------

    def channel(self) -> str:
        return U.normalize_channel(self.edit_channel.text())

    def _remember_channel(self):
        ch = self.channel()
        if ch != self._channel:
            self._channel = ch
            if self._on_channel_change:
                self._on_channel_change(ch)

    # ---- check -----------------------------------------------------------

    def check(self):
        self._remember_channel()
        ch = self.channel()
        if not ch:
            self.lbl_status.setText(
                "Set a channel first — a folder everyone's copy can reach, or "
                "<code>github:owner/repo</code>.")
            return
        self.btn_check.setEnabled(False)
        self.btn_install.setEnabled(False)
        self.notes.setVisible(False)
        self.lbl_status.setText(f"Checking {U.describe_channel(ch)} …")
        self._worker = CheckWorker(ch)
        self._worker.done.connect(self._checked)
        self._worker.start()

    def _checked(self, rel, err):
        self.btn_check.setEnabled(True)
        if err:
            self.lbl_status.setText("Check failed.")
            QMessageBox.warning(self, "Update check", err)
            return
        if rel is None:
            self._rel = None
            self.lbl_status.setText(
                f"robotrack {U.current_version()} is up to date.")
            return

        ok, why = U.can_apply(rel)
        self._rel = rel
        size = f"{rel.size / 1024 / 1024:.1f} MB" if rel.size > 1024 * 1024 \
            else f"{max(rel.size, 0) / 1024:.0f} KB"
        kind = "code patch" if rel.is_code else "full installer"
        self.lbl_status.setText(
            f"<b>Version {rel.version}</b> is available — {kind}, {size}."
            + ("" if ok else f"<br><span>{why}</span>"))
        if rel.notes:
            self.notes.setPlainText(rel.notes)
            self.notes.setVisible(True)
        self.btn_install.setEnabled(ok)

    # ---- install ---------------------------------------------------------

    def install(self):
        if self._rel is None:
            return
        ok, why = U.can_apply(self._rel)
        if not ok:
            QMessageBox.warning(self, "Update", why)
            return
        self.btn_install.setEnabled(False)
        self.btn_check.setEnabled(False)
        self.bar.setVisible(True)
        self.bar.setRange(0, 0)
        self.lbl_status.setText(f"Downloading {self._rel.version} …")
        self._worker = DownloadWorker(self._rel)
        self._worker.progress.connect(self._progress)
        self._worker.done.connect(self._downloaded)
        self._worker.start()

    def _progress(self, done, total):
        if total:
            self.bar.setRange(0, total)
            self.bar.setValue(done)

    def _downloaded(self, path, err):
        self.bar.setVisible(False)
        self.btn_check.setEnabled(True)
        if err or path is None:
            self.btn_install.setEnabled(True)
            self.lbl_status.setText("Download failed. Nothing was changed.")
            QMessageBox.critical(self, "Update", err or "Download failed.")
            return

        rel = self._rel
        installed = False
        try:
            if rel.is_code:
                U.apply_code_update(Path(path), rel)
            else:
                U.apply_full_update(Path(path))
            installed = True
        except U.UpdateError as exc:
            self.btn_install.setEnabled(True)
            QMessageBox.critical(self, "Update", str(exc))
        except Exception:
            self.btn_install.setEnabled(True)
            QMessageBox.critical(self, "Update", traceback.format_exc())
        finally:
            if rel.is_code:
                Path(path).unlink(missing_ok=True)
        if not installed:
            return

        # Installing and restarting are two separate outcomes, and conflating
        # them is what made a working update look like a broken one: the patch
        # landed correctly every time, the self-restart quietly did not, and the
        # program carried on as the old version with nothing said. Now the
        # install is reported on its own terms, and the restart is an attempt
        # made afterwards whose failure is only an inconvenience.
        # Recorded on the dialog, not announced by it.
        #
        # This used to close the dialog and then ask a queued signal to carry
        # the restart request out of it. That request has to survive a modal
        # loop unwinding, a dialog being accepted, and the last Python reference
        # to that dialog going out of scope in the caller -- and the observed
        # behaviour was the progress bar reaching 100%, the window vanishing,
        # and nothing at all afterwards: no restart, no error, no message. A
        # request that can be dropped in flight is the wrong shape for the one
        # step the whole update depends on.
        #
        # The caller reads this field after exec() returns, which is ordinary
        # synchronous code in the main window, after every loop has unwound.
        self.installed_version = rel.version
        self.lbl_status.setText(f"Version {rel.version} installed.")
        self.accept()
