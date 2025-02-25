from __future__ import annotations

import numpy as np
import wgpu.gui.jupyter, wgpu.gui.auto

from ..drawreason import DrawReason

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..visualizer import Visualizer


from wgpu.gui.base import WgpuCanvasBase


class VisualizerCanvasBase:
    def __init__(self, *args, **kwargs):
        self._visualizer : Visualizer = kwargs.pop("visualizer")

        self._last_x = 0
        self._last_y = 0
        # The below are dummy values that will be updated by the initial resize event
        self.width_physical, self.height_physical = 640,480
        self.pixel_ratio = 1
        self.split_view = False  # New flag for split-screen mode

        super().__init__(*args, **kwargs)

    def handle_event(self, event): #Handle UI interactions like mouse movement, clicks, and resizing. If split view is enabled, ensure interactions apply to the correct visualization.
    
        width_half = self.width_physical // 2  # Divide screen width in half

        if event["event_type"] == "pointer_move":
            if self.split_view:
                # Determine which half of the screen the mouse is in
                if event["x"] < width_half:
                    # Left side (Primary visualization)
                    self._visualizer.hover(event["x"] - self._last_x, event["y"] - self._last_y)
                else:
                    # Right side (Comparison visualization)
                    dx_adjusted = event["x"] - width_half - self._last_x  # Adjust X for right view
                    self._visualizer.hover(dx_adjusted, event["y"] - self._last_y)

            else:
                # Single visualization mode (Default)
                self._visualizer.hover(event["x"] - self._last_x, event["y"] - self._last_y)

            # Update last mouse position
            self._last_x = event["x"]
            self._last_y = event["y"]

        elif event["event_type"] == "pointer_down":
            # Handle mouse clicks (e.g., rotation, dragging)
            if self.split_view:
                if event["x"] < width_half:
                    self._visualizer.drag(event["x"] - self._last_x, event["y"] - self._last_y)  # Left view
                else:
                    dx_adjusted = event["x"] - width_half - self._last_x  # Right view
                    self._visualizer.drag(dx_adjusted, event["y"] - self._last_y)
            else:
                self._visualizer.drag(event["x"] - self._last_x, event["y"] - self._last_y)

        elif event["event_type"] == "wheel":
            # Handle zooming via mouse wheel
            zoom_factor = np.exp(event["dy"] / 1000)
            if self.split_view:
                if event["x"] < width_half:
                    self._visualizer.scale *= zoom_factor  # Left visualization
                else:
                    self._visualizer.scale *= zoom_factor  # Right visualization (optional sync)
            else:
                self._visualizer.scale *= zoom_factor  # Single view zoom

        elif event["event_type"] == "key_up":
            # Handle keyboard shortcuts
            if event["key"] == "s":
                self._visualizer.save()
            elif event["key"] == "r":
                self._visualizer.vmin_vmax_is_set = False
                self._visualizer.invalidate()
            elif event["key"] == "h":
                self._visualizer.reset_view()

        elif event["event_type"] == "resize":
            # Ensure both views adjust correctly on resize
            self.resize_complete(event["width"], event["height"], event["pixel_ratio"])
            if self.split_view:
                self._visualizer.invalidate(DrawReason.PRESENTATION_CHANGE)  # Force redraw in split mode

        elif event["event_type"] == "double_click":
            # Optional: Define behavior for double-clicks
            self.double_click(event["x"], event["y"])

        elif event["event_type"] == "pointer_up":
            # Handle mouse release
            self.release_drag()

        # Pass event handling to the base class
        super().handle_event(event)


    def hover(self, dx, dy): # Defines an event for mouse hovering
        # print(f"Canvas Event: dx={dx}, dy={dy}") # debugging
        self._visualizer.hover(dx, dy) # calls hover function from visualizer.py
        self._visualizer.invalidate() # updates visualization

    def drag(self, dx, dy):
        self._visualizer.rotate(dx*0.01, dy*0.01)

    def shift_drag(self, dx, dy):
        biggest_dimension = max(self.width_physical, self.height_physical)

        displacement = 2.*self.pixel_ratio*np.array([dx, -dy, 0], dtype=np.float32) / biggest_dimension * self._visualizer.scale
        self._visualizer.position_offset += self._visualizer.rotation_matrix.T @ displacement

        self._visualizer.display_status("centre = [{:.2f}, {:.2f}, {:.2f}]".format(*self._visualizer._sph.position_offset))

        self._visualizer.crosshairs_visible = True


    def key_up(self, key):
        if key=='s':
            self._visualizer.save()
        elif key=='r':
            self._visualizer.vmin_vmax_is_set = False
            self._visualizer.invalidate()
        elif key=='h':
            self._visualizer.reset_view()

    def mouse_wheel(self, delta_x, delta_y):
        if isinstance(self, wgpu.gui.jupyter.JupyterWgpuCanvas):
            # scroll events are much smaller from the web browser, for
            # some reason, compared with native windowing
            delta_y *= 10
            delta_x *= 10

        self._visualizer.scale*=np.exp(delta_y/1000)

    def release_drag(self):
        if self._visualizer.crosshairs_visible:
            self._visualizer.crosshairs_visible = False
            self._visualizer.invalidate()


    def resize(self, *args):
        # putting this here as a reminder that the resize method must be passed to the base class
        super().resize(*args)
    def resize_complete(self, width, height, pixel_ratio=1):
        self.width_physical = int(width*pixel_ratio)
        self.height_physical = int(height*pixel_ratio)
        self.pixel_ratio = pixel_ratio

        if self.split_view:
            self._visualizer.invalidate(DrawReason.PRESENTATION_CHANGE)  # Force redraw for split mode

    def double_click(self, x, y):
        pass

    @classmethod
    def call_later(cls, delay, fn, *args):
        raise NotImplementedError()

    def enable_split_view(self): #Enable side-by-side rendering for two visualizations.
        self.split_view = True
        self.request_draw()

    def disable_split_view(self): #Disable side-by-side rendering (single view mode).
        self.split_view = False
        self.request_draw()




# Now we are going to select a specific backend
#
# we don't use wgpu.gui.auto directly because it prefers the glfw backend over qt
# whereas we want to use qt
#
# Note also that is_jupyter as implemented fails to distinguish correctly if we are
# running inside a kernel that isn't attached to a notebook. There doesn't seem to
# be any way to distinguish this, so we live with it for now.

def is_jupyter():
    """Determine whether the user is executing in a Jupyter Notebook / Lab.

    This has been pasted from an old version of wgpu.gui.auto.is_jupyter; the function was removed"""
    from IPython import get_ipython
    try:
        ip = get_ipython()
        if ip is None:
            return False
        if ip.has_trait("kernel"):
            return True
        else:
            return False
    except NameError:
        return False


if is_jupyter():
    from .jupyter import VisualizerCanvas
else:
    from .qt import VisualizerCanvas

