from __future__ import annotations

import logging
import numpy as np
import time
import wgpu
import math
import time
import matplotlib

from contextlib import contextmanager

from . import config
from . import canvas
from . import colormap
from . import multiresolution_sph, sph, periodic_sph
from . import colorbar
from . import text
from . import scalebar
from . import loader
from . import util
from . import line
from . import simcube
from . import view_synchronizer
from .drawreason import DrawReason
from .sphere import SphereOverlay  # assuming you save the SphereOverlay class in sphere.py


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
class VisualizerBase:
    colorbar_aspect_ratio = config.COLORBAR_ASPECT_RATIO
    show_status = True
    device = None # device will be shared across all instances

    def __init__(self, data_loader_class = loader.TestDataLoader, data_loader_args = (),
                 *, render_resolution = config.DEFAULT_RESOLUTION, periodic_tiling = False,
                 colormap_name = config.DEFAULT_COLORMAP, canvas_class = canvas.VisualizerCanvas):
        self.split_screen_enabled = False  # Flag to track split-screen state
        self._colormap_name = colormap_name
        self._render_resolution = render_resolution
        self.crosshairs_visible = False

        self._prevent_sph_rendering = False # when True, prevents the sph from rendering, to ensure quick screen updates
        self.vmin_vmax_is_set = False

        self.show_colorbar = True
        self.show_scalebar = True
        
        # initialize mouse position attributes
        self.last_mouse_x = 0
        self.last_mouse_y = 0

        # initialize mouse absolute position attributes
        self.abs_x = 0
        self.abs_y = 0

        self.show_sphere = False
        self.canvas = canvas_class(visualizer=self, title="topsy")

        self._setup_wgpu()

        self._sphere_overlay = SphereOverlay(self, position=(0, 0, 0), radius=2000) # previous radius was too tiny, probably need to set default radius proportional to the simulation size

        self.data_loader = data_loader_class(self.device, *data_loader_args)

        self.periodicity_scale = self.data_loader.get_periodicity_scale()

        self._colormap = colormap.Colormap(self, weighted_average = False)
        self._periodic_tiling = periodic_tiling

        # maintain original min/max values for color shift to reference (instead of compounding based on updated vmin/vmax)
        self.original_vmin = None # by default, vmin/vmax are set to 0/1 -> set to None initially, wait for autorange_vmin_vmax to set the colormap's original vmin/vmax
        self.original_vmax = None
        
        self._colormap_exponent = 1.0 # exponent to shift vmin/vmax by (1.0 means linear shift)

        if periodic_tiling:
            self._sph = periodic_sph.PeriodicSPH(self, self.render_texture)
        else:
            self._sph = sph.SPH(self, self.render_texture)
        #self._sph = multiresolution_sph.MultiresolutionSPH(self, self.render_texture)

        self._last_status_update = 0.0
        self._status = text.TextOverlay(self, "topsy", (-0.9, 0.9), 80, color=(1, 1, 1, 1))

        self._colorbar = colorbar.ColorbarOverlay(self, 0.0, 1.0, self.colormap_name, "TODO")
        self._scalebar = scalebar.ScalebarOverlay(self)
        # self._colorslider = colorslider.ColorSliderOverlay(self)

        self._crosshairs = line.Line(self,
                                     [(-1, 0,0,0), (1, 0,0,0),
                                      (200,200,0,0),
                                      (0, 1, 0, 0), (0, -1, 0, 0)],
                                     (1, 1, 1, 0.3) # color
                                     , 10.0)
        self._cube = simcube.SimCube(self, (1, 1, 1, 0.3), 10.0)

        self._render_timer = util.TimeGpuOperation(self.device)

        self.invalidate(DrawReason.INITIAL_UPDATE)

        self.projection_matrix = self.create_projection_matrix()
        self.view_matrix = np.eye(4)

    def _setup_wgpu(self):
        self.adapter: wgpu.GPUAdapter = wgpu.gpu.request_adapter(power_preference="high-performance")
        if self.device is None:
            max_buffer_size = self.adapter.limits['max_buffer_size']
            # on some systems, this is 2^64 which can lead to overflows
            if max_buffer_size > 2**63:
                max_buffer_size = 2**63
            type(self).device: wgpu.GPUDevice = self.adapter.request_device(
                required_features=["TextureAdapterSpecificFormatFeatures", "float32-filterable"],
                required_limits={"max_buffer_size": max_buffer_size})
        self.context: wgpu.GPUCanvasContext = self.canvas.get_context()
        self.canvas_format = self.context.get_preferred_format(self.adapter)
        if self.canvas_format.endswith("-srgb"):
            # matplotlib colours aren't srgb. It might be better to convert
            # but for now, just stop the canvas being srgb
            self.canvas_format = self.canvas_format[:-5]
        self.context.configure(device=self.device, format=self.canvas_format)
        self.render_texture: wgpu.GPUTexture = self.device.create_texture(
            size=(self._render_resolution, self._render_resolution, 1),
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT |
                  wgpu.TextureUsage.TEXTURE_BINDING |
                  wgpu.TextureUsage.COPY_SRC,
            format=wgpu.TextureFormat.rg32float,
            label="sph_render_texture",
        )

    def invalidate(self, reason=DrawReason.CHANGE):
        # NB no need to check if we're already pending a draw - wgpu.gui does that for us
        self.canvas.request_draw(lambda: self.draw(reason))

    def rotate(self, x_angle, y_angle):
        dx_rotation_matrix = self._x_rotation_matrix(x_angle)
        dy_rotation_matrix = self._y_rotation_matrix(y_angle)
        self.rotation_matrix = dx_rotation_matrix @ dy_rotation_matrix @ self.rotation_matrix
    
    # this function is used to shift the vmin/vmax values of the colormap based on the value (exponent) of the UI contrast slider    
    def set_colormap_exponent(self, exponent):
        vmin = self.original_vmin
        vmax = self.original_vmax

        span = vmax - vmin # range between original vmin and vmax

        # make sure exponent is within bounds
        exponent = np.clip(exponent, 0.5, 1.5)

        # boost dark areas (right side of slider)
        if exponent > 1.0:
            factor = (exponent - 1.0) * span * 0.8 # when on right side of slider, we are shifting the vmax downwards to boost the dark areas
            new_vmin = vmin
            new_vmax = vmax - factor
        # boost bright areas (left side of slider)
        elif exponent < 1.0: # use exponent to shift vmin/vmax from original values
            factor = (1.0 - exponent) * span * 0.8  # when on left side of slider, we are shifting the vmin upwards to boost the bright areas
            new_vmin = vmin + factor
            new_vmax = vmax
        # if exponent is 1, maintain original vmin/vmax values
        else:
            new_vmin = vmin
            new_vmax = vmax

        # update vmin and vmax
        self.vmin = new_vmin
        self.vmax = new_vmax
        self.invalidate(DrawReason.CHANGE)

    def set_colormap(self, cmap: matplotlib.colors.Colormap): # set colormap to a matplotlib colormap
        lut = cmap(np.linspace(0, 1, 256))[:, :3] # to use in topsy we have to convert the colormap format
        current_quantity = self.quantity_name
        if current_quantity is not None:
            self.data_loader.quantity_name = None
        topsy_cmap = colormap.Colormap(self, weighted_average=False) # create topsy colormap object
        topsy_cmap.set_custom_lut(lut) # set the custom LUT to the topsy colormap
        self._colormap = topsy_cmap

        self._colormap_name = "__custom__" # set the colormap name to custom

        # for colorbar - autorange vmin/vmax and store them
        self._colormap.autorange_vmin_vmax()
        # self.vmin_vmax_is_set = True
        self.original_vmin = self._colormap.vmin
        self.original_vmax = self._colormap.vmax
        self.vmin_vmax_is_set = True

        # set colorbar
        self._colorbar = colorbar.ColorbarOverlay(
        self,
        self._colormap.vmin,
        self._colormap.vmax,
        self._colormap,
        self._get_colorbar_label()
    )
        
        if current_quantity is not None:
            self.data_loader.quantity_name = current_quantity

            self._colormap = colormap.Colormap(self, weighted_average=True) # create topsy colormap object
            self._colormap.set_custom_lut(lut)

            self.vmin_vmax_is_set = False
            self._colormap.autorange_vmin_vmax()
            self.original_vmin = self._colormap.vmin
            self.original_vmax = self._colormap.vmax
            self.vmin_vmax_is_set = True

            self._colorbar = colorbar.ColorbarOverlay(
                self,
                self._colormap.vmin,
                self._colormap.vmax,
                self._colormap,
                self._get_colorbar_label()
            )

        self.invalidate() # update the colormap and colorbar

    @property
    def rotation_matrix(self):
        return self._sph.rotation_matrix

    @rotation_matrix.setter
    def rotation_matrix(self, value):
        self._sph.rotation_matrix = value
        self.invalidate()

    @property
    def colormap_name(self):
        return self._colormap_name

    @colormap_name.setter
    def colormap_name(self, value):
        self._colormap_name = value
        self._reinitialize_colormap_and_bar()
        self.invalidate(reason=DrawReason.PRESENTATION_CHANGE)

    @property
    def position_offset(self):
        return self._sph.position_offset

    @position_offset.setter
    def position_offset(self, value):
        self._sph.position_offset = value
        self.invalidate()

    def reset_view(self):
        self._sph.rotation_matrix = np.eye(3)
        self.scale = config.DEFAULT_SCALE
        self._sph.position_offset = np.zeros(3)

    def shrink_sphere(self):
        if hasattr(self, "_sphere_overlay"):
            self._sphere_overlay.set_radius(self._sphere_overlay._radius * 0.9)
            self.invalidate()

    def expand_sphere(self):
        if hasattr(self, "_sphere_overlay"):
            self._sphere_overlay.set_radius(self._sphere_overlay._radius * 1.1)
            self.invalidate()

    def move_sphere(self, dx=0.0, dy=0.0, dz=0.0):
        if not hasattr(self, "_sphere_overlay"):
            return
        current_pos = self._sphere_overlay.position
        new_pos = current_pos + np.array([dx, dy, dz], dtype=np.float32)
        self._sphere_overlay.set_position_and_radius(new_pos, self._sphere_overlay._radius)
        self.invalidate()

    def show_average_properties_in_sphere(self):
        """Find particles in sphere and show averaged properties in the popup."""
        if not (self.show_sphere and hasattr(self, "_sphere_overlay")):
            return

        center = self._sphere_overlay._position
        radius = self._sphere_overlay._radius
        particles = self.find_particles_in_sphere(center, radius)
        print(f"[SPHERE] Found {len(particles)} particles inside sphere")
        print(particles)

        if particles.shape[0] > 0:
            avg_props = self.get_average_properties(particles)

            if hasattr(self.canvas, "popup"):
                self.canvas.popup.update_info(avg_props)

    @property
    def scale(self):
        """Return the scalefactor from kpc to viewport coordinates. Viewport will therefore be 2*scale wide."""
        return self._sph.scale
    @scale.setter
    def scale(self, value):
        self._sph.scale = value
        self.invalidate()

    @property
    def quantity_name(self):
        """The name of the quantity being visualised, or None if density projection."""
        return self.data_loader.quantity_name

    @property
    def averaging(self):
        """True if the quantity being visualised is a weighted average, False if it is a mass projection."""
        return self.data_loader.quantity_name is not None

    @quantity_name.setter
    def quantity_name(self, value):
        has_custom_colormap = self._colormap_name == "__custom__" # check if colormap is custom
        custom_lut = None
        if has_custom_colormap and hasattr(self._colormap, "custom_lut"):
            custom_lut = self._colormap.custom_lut.copy() # copy the custom LUT to avoid modifying the original one

        if value is not None:
            # see if we can get it. Assume it'll be cached, so this won't waste time.
            try:
                self.data_loader.get_named_quantity(value)
            except Exception as e:
                raise ValueError(f"Unable to get quantity named '{value}'") from e

        self.data_loader.quantity_name = value
        self.vmin_vmax_is_set = False
        self._reinitialize_colormap_and_bar()

        if has_custom_colormap and custom_lut is not None: # if there was a custom LUT before quantity was changed, reapply the LUT to the initialized colormap
            self._colormap.set_custom_lut(custom_lut) # set the custom LUT to the colormap

        self.invalidate()

    def _reinitialize_colormap_and_bar(self):
        if self._colormap_name == "__custom__": # handle custom colormap
            if hasattr(self._colormap, "custom_lut") and getattr(self._colormap, "use_custom_lut", False): # if the colormap has attribute custom_lut and use_custom_lut is false, then we need to set the custom LUT
                lut = self._colormap.custom_lut # custom LUT is a numpy array of shape (256,3)
                vmin, vmax, log_scale = self.vmin, self.vmax, self.log_scale
                self._colormap = colormap.Colormap(self, weighted_average=self.quantity_name is not None)
                self._colormap.set_custom_lut(lut) # set the custom LUT to the colormap
                if self.vmin_vmax_is_set:
                    self._colormap.vmin = vmin
                    self._colormap.vmax = vmax
                    self._colormap.log_scale = log_scale
                self._colorbar = colorbar.ColorbarOverlay(self, self.vmin, self.vmax, self._colormap, self._get_colorbar_label()) # set the colorbar to the custom colormap object
            return

        # handle standard colormaps
        vmin, vmax, log_scale = self.vmin, self.vmax, self.log_scale
        self._colormap = colormap.Colormap(self, weighted_average=self.quantity_name is not None)
        if self.vmin_vmax_is_set:
            self._colormap.vmin = vmin
            self._colormap.vmax = vmax
            self._colormap.log_scale = log_scale
        self._colorbar = colorbar.ColorbarOverlay(self, self.vmin, self.vmax, self.colormap_name,
                                                  self._get_colorbar_label())

    def _get_colorbar_label(self):
        label = self.data_loader.get_quantity_label()
        if self._colormap.log_scale:
            label = r"$\log_{10}$ " + label
        return label

    @staticmethod
    def _y_rotation_matrix(angle):
        return np.array([[1, 0, 0],
                         [0, np.cos(angle), -np.sin(angle)],
                         [0, np.sin(angle), np.cos(angle)]])

    @staticmethod
    def _x_rotation_matrix(angle):
        return np.array([[np.cos(angle), 0, np.sin(angle)],
                         [0, 1, 0],
                         [-np.sin(angle), 0, np.cos(angle)]])

    def _check_whether_inactive(self):
        if time.time()-self._last_lores_draw_time>config.FULL_RESOLUTION_RENDER_AFTER*0.95:
            self._last_lores_draw_time = np.inf # prevent this from being called again
            self.invalidate(reason=DrawReason.REFINE)

    @contextmanager
    def prevent_sph_rendering(self):
        self._prevent_sph_rendering = True
        try:
            yield
        finally:
            self._prevent_sph_rendering = False

    def draw(self, reason, target_texture_view=None):


        if reason == DrawReason.REFINE or reason == DrawReason.EXPORT:
            self._sph.downsample_factor = 1

        if reason!=DrawReason.PRESENTATION_CHANGE and (not self._prevent_sph_rendering):
            ce_label = "sph_render"
            # labelling this is useful for understanding performance in macos instruments
            if self._sph.downsample_factor>1:
                ce_label += f"_ds{self._sph.downsample_factor:d}"
            else:
                ce_label += "_fullres"

            command_encoder : wgpu.GPUCommandEncoder = self.device.create_command_encoder(label=ce_label)
            self._sph.encode_render_pass(command_encoder)

            with self._render_timer:
                self.device.queue.submit([command_encoder.finish()])

        if not self.vmin_vmax_is_set:
            logger.info("Setting vmin/vmax - testing")
            self._colormap.autorange_vmin_vmax()
            self.vmin_vmax_is_set = True

            self.original_vmin = self._colormap.vmin # set original vmin/vmax values for visualizer to use -> without this, visualizer uses default vmin/vmax (0,1) to reference for the colormap shifting limits
            self.original_vmax = self._colormap.vmax

            self._refresh_colorbar()

        command_encoder = self.device.create_command_encoder()

        if target_texture_view is None:
            target_texture_view = self.canvas.get_context().get_current_texture().create_view()


        self._colormap.encode_render_pass(command_encoder, target_texture_view)
        if self.show_colorbar:
            self._colorbar.encode_render_pass(command_encoder, target_texture_view)
        if self.show_scalebar:
            self._scalebar.encode_render_pass(command_encoder, target_texture_view)
        if self.show_sphere:
            self._sphere_overlay.encode_render_pass(command_encoder, target_texture_view)
        if self.crosshairs_visible:
            self._crosshairs.encode_render_pass(command_encoder, target_texture_view)
        if self._periodic_tiling:
            self._cube.encode_render_pass(command_encoder, target_texture_view)

        if reason == DrawReason.REFINE:
            self.display_status("Full-res render took {:.2f} s".format(self._render_timer.last_duration, timeout=0.1))

        if self.show_status:
            self._update_and_display_status(command_encoder, target_texture_view)

        self.device.queue.submit([command_encoder.finish()])



        if reason != DrawReason.PRESENTATION_CHANGE and reason != DrawReason.EXPORT and (not self._prevent_sph_rendering):
            if self._sph.downsample_factor>1:
                self._last_lores_draw_time = time.time()
                self.canvas.call_later(config.FULL_RESOLUTION_RENDER_AFTER, self._check_whether_inactive)
            elif self._render_timer.last_duration>1/config.TARGET_FPS and self._sph.downsample_factor==1:
                # this will affect the NEXT frame, not this one!
                self._sph.downsample_factor = int(np.floor(float(config.TARGET_FPS)*self._render_timer.last_duration))

    @property
    def vmin(self):
        return self._colormap.vmin

    @property
    def vmax(self):
        return self._colormap.vmax

    @vmin.setter
    def vmin(self, value):
        self._colormap.vmin = value
        self.vmin_vmax_is_set = True
        self._refresh_colorbar()
        self.invalidate()

    @vmax.setter
    def vmax(self, value):
        self._colormap.vmax = value
        self.vmin_vmax_is_set = True
        self._refresh_colorbar()
        self.invalidate()

    @property
    def log_scale(self):
        return self._colormap.log_scale

    @log_scale.setter
    def log_scale(self, value):
        self._colormap.log_scale = value
        self._refresh_colorbar()
        self.invalidate()

    def _refresh_colorbar(self):
        self._colorbar.vmin = self._colormap.vmin
        self._colorbar.vmax = self._colormap.vmax
        self._colorbar.label = self._get_colorbar_label()
        self._colorbar.update()

    def sph_clipspace_to_screen_clipspace_matrix(self):
        aspect_ratio = self.canvas.width_physical / self.canvas.height_physical
        if aspect_ratio>1:
            y_squash = aspect_ratio
            x_squash = 1.0
        elif aspect_ratio<1:
            y_squash = 1.0
            x_squash = 1.0/aspect_ratio
        else:
            x_squash = 1.0
            y_squash = 1.0

        matr = np.eye(4, dtype=np.float32)
        matr[0,0] = x_squash
        matr[1,1] = y_squash
        return matr



    def display_status(self, text, timeout=0.5):
        self._override_status_text = text
        self._override_status_text_until = time.time()+timeout

    def _update_and_display_status(self, command_encoder, target_texture_view):
        now = time.time()
        if hasattr(self, "_override_status_text_until") and now<self._override_status_text_until:
            if self._status.text!=self._override_status_text and now-self._last_status_update>config.STATUS_LINE_UPDATE_INTERVAL_RAPID:
                self._status.text = self._override_status_text
                self._last_status_update = now
                self._status.update()

        elif now - self._last_status_update > config.STATUS_LINE_UPDATE_INTERVAL:
            self._last_status_update = now
            self._status.text = f"${1.0 / self._render_timer.running_mean_duration:.0f}$ fps"
            if self._sph.downsample_factor > 1:
                self._status.text += f", downsample={self._sph.downsample_factor:d}"

            self._status.update()

        self._status.encode_render_pass(command_encoder, target_texture_view)

    def get_sph_image(self) -> np.ndarray:
        im = self.device.queue.read_texture({'texture':self.render_texture, 'origin':(0, 0, 0)},
                                            {'bytes_per_row':8*self._render_resolution},
                                            (self._render_resolution, self._render_resolution, 1))
        np_im = np.frombuffer(im, dtype=np.float32).reshape((self._render_resolution, self._render_resolution, 2))
        if self.averaging:
            im = np_im[:,:,1]/np_im[:,:,0]
        else:
            im = np_im[:,:,0]
        return im

    def get_presentation_image(self) -> np.ndarray:
        texture = self.context.get_current_texture()
        size = texture.size
        bytes_per_pixel = 4 # NB this might be wrong in principle!
        data = self.device.queue.read_texture(
            {
                "texture": texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            {
                "offset": 0,
                "bytes_per_row": bytes_per_pixel * size[0],
                "rows_per_image": size[1],
            },
            size,
        )

        return np.frombuffer(data, np.uint8).reshape(size[1], size[0], 4)


    def save(self, filename='output.pdf'):
        image = self.get_sph_image()
        import matplotlib.pyplot as p
        fig = p.figure()
        p.clf()
        p.set_cmap(self.colormap_name)
        extent = np.array([-1., 1., -1., 1.])*self.scale
        if self._colormap.log_scale:
            image = np.log10(image)

        p.imshow(image,
                 vmin=self._colormap.vmin,
                 vmax=self._colormap.vmax,
                 extent=extent)
        p.xlabel("$x$/kpc")
        p.colorbar().set_label(self._colorbar.label)
        p.savefig(filename)
        p.close(fig)

    def show(self, force=False):
        from wgpu.gui import jupyter
        if isinstance(self.canvas, jupyter.WgpuCanvas):
            return self.canvas
        else:
            from wgpu.gui import qt # can only safely import this if we think we're running in a qt environment
            assert isinstance(self.canvas, qt.WgpuCanvas)
            self.canvas.show()
            if force or not util.is_inside_ipython():
                qt.run()
            elif not util.is_ipython_running_qt_event_loop():
                # is_inside_ipython_console must be True; if it were False, the previous branch would have run
                # instead.
                print("\r\nYou appear to be running from inside ipython, but the gui event loop is not running.\r\n"
                      "Please run %gui qt in ipython before calling show().\r\n"
                      "\r\n"
                      "Alternatively, if you do not want to continue interacting with ipython while the\r\n"
                      "visualizer is running, you can call show(force=True) to run the gui without access\r\n"
                      "to the ipython console until you close the visualizer window.\r\n\r\n"
                      )
        #else:
        #    raise RuntimeError("The wgpu library is using a gui backend that topsy does not recognize")

    def create_projection_matrix(self, fov=60, aspect_ratio=1.0, near=0.1, far=1000.0):
        """Creates a simple perspective projection matrix for 3D coordinate conversion."""
        f = 1.0 / np.tan(np.radians(fov) / 2.0)
        return np.array([
            [f / aspect_ratio, 0,  0,                                  0],
            [0,               f,  0,                                  0],
            [0,               0,  (far + near) / (near - far),       -1],
            [0,               0,  (2 * far * near) / (near - far),    0]
        ], dtype=np.float32)
    
    def screen_to_world(self, x, y, screen_width, screen_height):
        """Convert 2D screen space (x, y) to a 3D world space ray."""
        
        import numpy as np  # Ensure numpy is imported

        # Step 1: Normalize screen coordinates to NDC (-1 to 1 range)
        ndc_x = (2.0 * x) / screen_width - 1.0
        ndc_y = 1.0 - (2.0 * y) / screen_height  # Flip Y-axis for OpenGL-style coordinates
        
        # Step 2: Convert to homogeneous clip space
        clip_coords = np.array([ndc_x, ndc_y, -1.0, 1.0])  # Assume looking into -Z direction

        # Step 3: Convert to eye space using inverse projection matrix
        inv_projection = np.linalg.inv(self.projection_matrix)  
        eye_coords = inv_projection @ clip_coords
        eye_coords = np.array([eye_coords[0], eye_coords[1], -1.0, 0.0])  # Reset depth

        # Step 4: Convert to world space using inverse view matrix
        inv_view = np.linalg.inv(self.view_matrix)  
        world_ray = inv_view @ eye_coords
        world_ray = world_ray[:3]  # Extract 3D direction

        # Normalize the ray direction
        world_ray = world_ray / np.linalg.norm(world_ray)
        
        return world_ray
    
    def get_particle_positions(self):
        """Retrieve all particle positions from the loaded simulation."""
        if not hasattr(self, "data_loader"):
            print("Error: No data loader found.")
            return np.array([])

        # Assuming `self.data_loader` has a function to get positions
        return self.data_loader.get_positions()
    
    def find_nearest_particle(self, ray_origin, ray_direction):
        """Find the closest particle along the given ray."""
        import numpy as np

        positions = self.get_particle_positions()
        if positions.size == 0:
            print("No particle positions available.")
            return None

        # Step 1: Compute the vector from the ray origin to each particle
        ray_to_particles = positions - ray_origin

        # Step 2: Project this vector onto the ray direction
        projections = np.dot(ray_to_particles, ray_direction)

        # Step 3: Compute perpendicular distance from ray to particles
        closest_points = ray_origin + np.outer(projections, ray_direction)
        distances = np.linalg.norm(positions - closest_points, axis=1)

        # Step 4: Find the particle with the smallest perpendicular distance
        min_index = np.argmin(distances)
        min_distance = distances[min_index]

        # Step 5: Define a selection threshold (adjust as needed)
        selection_threshold = 0.05  # Adjust based on your simulation scale

        if min_distance < selection_threshold:
            print(f"Selected Particle {min_index} at {positions[min_index]} (Distance: {min_distance})")
            return positions[min_index]  # Return the selected particle position

        print("No particle selected.")
        return None

    def get_particle_properties(self, particle_position):
        """Retrieve and print all available properties of the selected particle dynamically."""
        import numpy as np

        if not hasattr(self, "data_loader"):
            print("Error: No data loader found.")
            return {}

        positions = self.get_particle_positions()
        if positions.size == 0:
            print("No particle positions available.")
            return {}

        # Find the index of the selected particle
        index = np.where((positions == particle_position).all(axis=1))[0]
        if index.size == 0:
            print("Error: Selected particle not found in dataset.")
            return {}

        index = index[0]  # Get the first match

        # Get all available properties in the dataset
        available_properties = list(self.data_loader.snapshot.keys())

        # Dynamically retrieve and store properties
        properties = {}
        for prop_name in available_properties:
            try:
                prop_value = self.data_loader.get_named_quantity(prop_name)[index]
                properties[prop_name.capitalize()] = prop_value
            except KeyError:
                continue  # Skip missing properties

        return properties
    
    def toggle_sphere_visibility(self, show):
        self.show_sphere = show
        self.invalidate()

    def get_average_properties(self, selected_positions):
        """Compute average mass, density, temperature for given positions."""
        pos_all = self.data_loader.get_positions()
        indices = np.flatnonzero((pos_all[:, None] == selected_positions).all(-1).any(1))

        props = {}
        for key in ["mass", "rho", "temp"]:
            try:
                values = self.data_loader.get_named_quantity(key)[indices]
                props[key.capitalize()] = float(np.mean(values))
            except Exception:
                continue
        return props



    def enable_split_view(self, second_canvas):
        #Enable split-screen mode.
        self.split_screen_enabled = True
        logger.info("✅ Split view has been enabled in the visualizer.")

        self.second_canvas = second_canvas # The second canvas to be used for split-screen mode


        self.invalidate(DrawReason.PRESENTATION_CHANGE)
        self.second_canvas.request_draw(lambda: self.draw(DrawReason.PRESENTATION_CHANGE, target_texture_view=self.second_canvas.get_context().get_current_texture().create_view()))

    def disable_split_view(self):
        #Disable split-screen mode.
        self.split_screen_enabled = False
        logger.info("🚫 Split view has been disabled in the visualizer.")
        self.invalidate(DrawReason.PRESENTATION_CHANGE)


class Visualizer(view_synchronizer.SynchronizationMixin, VisualizerBase):
    def find_particles_in_sphere(self, center, radius):
        # Get all particle positions from the data loader
        all_positions = self.data_loader.get_positions()  # <--- this!

        dists = np.linalg.norm(all_positions - center, axis=1)
        mask = dists <= radius

        logger.info(f"Found {np.sum(mask)} particles in sphere.")
        return all_positions[mask]

