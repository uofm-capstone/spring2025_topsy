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

        self._mouse_down_pos = None  # Store initial click position
        self._mouse_moved = False  # Track whether the mouse moved

        self._last_x = 0
        self._last_y = 0
        # The below are dummy values that will be updated by the initial resize event
        self.width_physical, self.height_physical = 640,480
        self.pixel_ratio = 1

        super().__init__(*args, **kwargs)

    def handle_event(self, event): # lea
        if event['event_type'] == 'pointer_down':
            # Store initial mouse position and reset movement tracking
            self._mouse_down_pos = (event['x'], event['y'])
            self._mouse_moved = False

        elif event['event_type'] == 'pointer_move':
            if len(event['buttons']) > 0:
                dx = abs(event['x'] - self._mouse_down_pos[0])
                dy = abs(event['y'] - self._mouse_down_pos[1])

                if dx > 5 or dy > 5:  # If moved significantly, it's a drag
                    self._mouse_moved = True

                if len(event['modifiers']) == 0:
                    self.drag(event['x'] - self._last_x, event['y'] - self._last_y)
                else:
                    self.shift_drag(event['x'] - self._last_x, event['y'] - self._last_y)
            # else:
            #     self.hover(event['x'] - self._last_x, event['y'] - self._last_y)

            self._last_x = event['x']
            self._last_y = event['y']

        elif event['event_type'] == 'wheel':
            self.mouse_wheel(event['dx'], event['dy'])

        elif event['event_type'] == 'key_up':
            self.key_up(event['key'])

        elif event['event_type'] == 'resize':
            self.resize_complete(event['width'], event['height'], event['pixel_ratio'])

        elif event['event_type'] == 'double_click':
            self.double_click(event['x'], event['y'])

        elif event['event_type'] == 'pointer_up':
            self.release_drag()
            '''
            # Only process selection if the mouse was not moved significantly (i.e., a click)
            if not self._mouse_moved:
                x, y = event['x'], event['y']
                screen_width, screen_height = self.width_physical, self.height_physical

                print(f"Mouse clicked at: ({x}, {y}) - Converting to 3D space...")

                ray_direction = self._visualizer.screen_to_world(x, y, screen_width, screen_height)
                ray_origin = np.array([0, 0, 0])
                nearest_particle = self._visualizer.find_nearest_particle(ray_origin, ray_direction)

                if nearest_particle is not None:
                    print(f"Selected Particle at {nearest_particle}")
                    properties = self._visualizer.get_particle_properties(nearest_particle)

                    # Ensure we call the popup from the correct place
                    if hasattr(self._visualizer.canvas, "popup"):
                        self._visualizer.canvas.popup.update_info(properties)
            '''
        else:
            pass
        super().handle_event(event)

    def drag(self, dx, dy):
        self._visualizer.rotate(dx*0.01, dy*0.01)

    def shift_drag(self, dx, dy):
        biggest_dimension = max(self.width_physical, self.height_physical)

        displacement = 2.*self.pixel_ratio*np.array([dx, -dy, 0], dtype=np.float32) / biggest_dimension * self._visualizer.scale
        self._visualizer.position_offset += self._visualizer.rotation_matrix.T @ displacement

        self._visualizer.display_status("centre = [{:.2f}, {:.2f}, {:.2f}]".format(*self._visualizer._sph.position_offset))

        self._visualizer.crosshairs_visible = True

    # when custom colormap is set, set "Custom" as the current cmap in the dropdown colormap menu
    def on_colormap_set_custom(self):
        pass
        # index = self._colormap_menu.findText("Custom")
        # if index == -1: # if "Custom" is not in the menu, add it
        #     self._colormap_menu.addItem("Custom")
        #     index = self._colormap_menu.findText("Custom")
        # self._colormap_menu.setCurrentIndex(index)

    def key_up(self, key):
        move_delta = 25  # Adjust for desired speed
        sphere = self._visualizer._sphere_overlay

        if key == 'v':
            self._visualizer.save()
        elif key == 'r':
            self._visualizer.vmin_vmax_is_set = False
            self._contrast_slider.setValue(100) # reset contrast slider to default 100
            self._visualizer.invalidate()
        elif key == 'h':
            self._visualizer.reset_view()
        elif key == 'f':
            self._visualizer.show_average_properties_in_sphere()
        elif key == '[':
            self._visualizer.shrink_sphere()
        elif key == ']':
            self._visualizer.expand_sphere()
        elif key == 'w':
            sphere.move_by((0, 0, -move_delta))
        elif key == 's':
            sphere.move_by((0, 0, move_delta))
        elif key == 'a':
            sphere.move_by((-move_delta, 0, 0))
        elif key == 'd':
            sphere.move_by((move_delta, 0, 0))
        elif key == 'q':
            sphere.move_by((0, move_delta, 0))
        elif key == 'e':
            sphere.move_by((0, -move_delta, 0))

        



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

    def double_click(self, x, y):
        pass

    @classmethod
    def call_later(cls, delay, fn, *args):
        raise NotImplementedError()





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

