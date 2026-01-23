#!/usr/bin/env python3
"""
Aria Launcher - Persistent menubar app to quick-launch Aria

This lightweight app stays in the macOS menu bar and lets you
start/stop Aria with a single click.
"""

import os
import signal
import subprocess
import sys
import time

import rumps

# Paths
ARIA_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(ARIA_DIR, "venv", "bin", "python")
ARIA_MODULE = "aria.main"
PID_FILE = os.path.join(ARIA_DIR, ".aria_pid")


class AriaLauncher(rumps.App):
    """Persistent menubar launcher for Aria."""

    def __init__(self):
        super().__init__(
            "Aria",
            title="Aria",
            quit_button=None  # We'll add our own quit button
        )

        self.aria_process = None
        self._check_existing_process()
        self._update_menu()

    def _check_existing_process(self):
        """Check if Aria is already running."""
        # Check PID file
        if os.path.exists(PID_FILE):
            try:
                with open(PID_FILE, "r") as f:
                    pid = int(f.read().strip())
                # Check if process is still running
                os.kill(pid, 0)  # Doesn't kill, just checks
                self.aria_process = pid
                return True
            except (ProcessLookupError, ValueError, FileNotFoundError):
                # Process not running, clean up PID file
                self._cleanup_pid_file()

        # Also check via pgrep
        try:
            result = subprocess.run(
                ["pgrep", "-f", "aria.main"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                if pids:
                    self.aria_process = int(pids[0])
                    self._save_pid(self.aria_process)
                    return True
        except Exception:
            pass

        return False

    def _save_pid(self, pid):
        """Save PID to file."""
        try:
            with open(PID_FILE, "w") as f:
                f.write(str(pid))
        except Exception:
            pass

    def _cleanup_pid_file(self):
        """Remove PID file."""
        try:
            if os.path.exists(PID_FILE):
                os.remove(PID_FILE)
        except Exception:
            pass

    def _is_running(self):
        """Check if Aria is currently running."""
        if self.aria_process:
            try:
                os.kill(self.aria_process, 0)
                return True
            except ProcessLookupError:
                self.aria_process = None
                self._cleanup_pid_file()

        # Double-check with pgrep
        return self._check_existing_process()

    def _update_menu(self):
        """Update menu based on Aria's status."""
        is_running = self._is_running()

        self.menu.clear()

        if is_running:
            self.title = "Aria (Running)"
            self.menu["Status"] = rumps.MenuItem("Status: Running", callback=None)
            self.menu["Status"].set_callback(None)
            self.menu["Toggle"] = rumps.MenuItem("Stop Aria", callback=self.toggle_aria)
        else:
            self.title = "Aria"
            self.menu["Status"] = rumps.MenuItem("Status: Stopped", callback=None)
            self.menu["Toggle"] = rumps.MenuItem("Start Aria", callback=self.toggle_aria)

        self.menu.add(rumps.separator)
        self.menu["Quit Launcher"] = rumps.MenuItem("Quit Launcher", callback=self.quit_launcher)

    @rumps.clicked("Start Aria")
    def toggle_aria(self, sender):
        """Start or stop Aria."""
        if self._is_running():
            self._stop_aria()
        else:
            self._start_aria()

        # Update menu after a short delay
        time.sleep(0.5)
        self._update_menu()

    def _start_aria(self):
        """Start Aria in background."""
        try:
            # Start Aria as a subprocess
            process = subprocess.Popen(
                [VENV_PYTHON, "-m", ARIA_MODULE],
                cwd=ARIA_DIR,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )

            self.aria_process = process.pid
            self._save_pid(process.pid)

            rumps.notification(
                "Aria",
                "Started",
                "Say 'Aria' or press Option+Space to activate"
            )
        except Exception as e:
            rumps.notification(
                "Aria",
                "Error",
                f"Failed to start: {str(e)[:50]}"
            )

    def _stop_aria(self):
        """Stop Aria."""
        try:
            if self.aria_process:
                # Try graceful termination first
                os.kill(self.aria_process, signal.SIGTERM)
                time.sleep(1)

                # Check if still running
                try:
                    os.kill(self.aria_process, 0)
                    # Still running, force kill
                    os.kill(self.aria_process, signal.SIGKILL)
                except ProcessLookupError:
                    pass

            # Also kill any other aria processes
            subprocess.run(
                ["pkill", "-9", "-f", "aria.main"],
                capture_output=True
            )

            self.aria_process = None
            self._cleanup_pid_file()

            rumps.notification(
                "Aria",
                "Stopped",
                "Aria has been stopped"
            )
        except Exception as e:
            rumps.notification(
                "Aria",
                "Error",
                f"Failed to stop: {str(e)[:50]}"
            )

    def quit_launcher(self, _):
        """Quit the launcher (but not Aria)."""
        rumps.quit_application()

    @rumps.timer(5)
    def check_status(self, _):
        """Periodically check Aria's status and update menu."""
        self._update_menu()


def main():
    """Run the Aria Launcher."""
    # Change to Aria directory
    os.chdir(ARIA_DIR)

    # Run the app
    AriaLauncher().run()


if __name__ == "__main__":
    main()
