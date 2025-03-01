from __future__ import annotations

import logging
import numpy as np
import time
import wgpu
import math
import time

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

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
class VisualizerBase:
    colorbar_aspect_ratio = config.COLORBAR_ASPECT_RATIO
    show_status = True
    device = None # device will be shared across all instances

    def __init__(self, data_loader_class = loader.TestDataLoader, data_loader_args = (),
                 *, render_resolution = config.DEFAULT_RESOLUTION, periodic_tiling = False,
                 colormap_name = config.DEFAULT_COLORMAP, canvas_class = canvas.VisualizerCanvas):
        self._colormap_name = colormap_name
        self._render_resolution = render_resolution
        self.crosshairs_visible = False

        self._prevent_sph_rendering = False # when True, prevents the sph from rendering, to ensure quick screen updates
        self.vmin_vmax_is_set = False

        self.show_colorbar = True
        self.show_scalebar = True

        # initialize mouse absolute position attributes
        self.abs_x = 0
        self.abs_y = 0

        self.canvas = canvas_class(visualizer=self, title="topsy")

        self._setup_wgpu()

        self.data_loader = data_loader_class(self.device, *data_loader_args)

        self.periodicity_scale = self.data_loader.get_periodicity_scale()

        self._colormap = colormap.Colormap(self, weighted_average = False)
        self._periodic_tiling = periodic_tiling

        # maintain original min/max values for color shift to reference (instead of compounding based on updated vmin/vmax)
        self.original_vmin = None # by default, vmin/vmax are set to 0/1 -> set to None initially, wait for autorange_vmin_vmax to set the colormap's original vmin/vmax
        self.original_vmax = None
        # self.original_vmin = 8 # hardcoded original vmin/vmax values for testing
        # self.original_vmax = 0.5

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

    # This function, for now, linearly shifts the color map's limits based on intensity of pixel under mouse so that the contrast within darker/lighter areas is enhanced
    # To do:
    # 1. tweak colormap shifting (using a better function to enhance contrast of darker/lighter areas, iord and some other color maps don't work)
    # 2. vmin/vmax capture is inconsistent, have to choose project density quantity a few times to get the correct vmin/vmax values
    # 3. make toggleable
    # 4. reduce amount of times hover triggers (and remove print statements)
    # 5. make light area differences more pronounced (dark areas are already decently pronounced)
    def hover(self, dx, dy):
        # update mouse position attributes
        self.abs_x += dx # calculates absolute position by adding change in position to last position (delta x, delta y)
        self.abs_y += dy

        # get rendered image sample data (using off-screen texture to avoid limitations of get_sph_image - can only get called once per frame)
        image = self.get_offscreen_image() # returns shape [height, width, 2]
        img_height = image.shape[0] # assign image height and width
        img_width = image.shape[1]
        # convert absolute mouse position to pixel coords in rendered image
        img_x = int(self.abs_x / self.canvas.width_physical * img_width)
        img_y = int(self.abs_y / self.canvas.height_physical * img_height)
        img_x = max(0, min(img_x, img_width - 1)) # make sure width and height within bounds
        img_y = max(0, min(img_y, img_height - 1))

        # maintain original vmin/vmax range
        # v_range = self.original_vmax - self.original_vmin

        # get pixel data from image using the coords
        pixel = image[img_y, img_x] # will be used to calculate intensity of the pixel
        # calculating intensity - how dark or light a pixel is so that the color map adjusts based on intensity
        intensity = pixel[0]

        # separate vmin/vmax shifting into separate function (currently sets vmin/vmax linearly based on intensity)
        new_vmin, new_vmax = self.apply_shift(intensity) # returns new vmin and vmax

        # setting new vmin/vmax (color scale limits) based on intensity
        # if intensity < 5000: # keep vmin/vmax the same if pixel is not intense enough (dark areas)
        #     new_vmin = max(self._colormap.vmin - 0.5, 0) # make sure vmin doesn't go below 0
        #     new_vmax = self._colormap.vmax
        # else: # bright areas
        #     new_vmin = self._colormap.vmin 
        #     new_vmax = min(self._colormap.vmax + 0.5, 12) # make sure vmax doesn't go above 12
            
        # print colormap change data
        print(f"absolute mouse positions: ({self.abs_x:.2f}, {self.abs_y:.2f}) // pixel coords: ({img_x}, {img_y}) // intensity: {intensity:.3f} // old vmin: {self._colormap.vmin:.3f} // old vmax: {self._colormap.vmax:.3f} // new vmin: {new_vmin:.3f} // new vmax: {new_vmax:.3f}")
        
        # update vmin/vmax
        self.vmin = new_vmin
        self.vmax = new_vmax

        # used for hover timeout (upon timeout, vmin/vmax will reset to original values)
        self._last_hover_time = time.time()

        self.invalidate(DrawReason.CHANGE) # mark that the visualizer needs to be updated
        
        self.canvas.call_later(1.0, self.reset_colormap_hover) # runs reset_colormap function after 1 second

    # shifts vmin/vmax based on intensity of pixel under mouse
    def apply_shift(self, intensity):
        if intensity < 10000: # for dark areas
            new_vmin = max(self._colormap.vmin - 2, 0) # make sure vmin doesn't go below 0
            new_vmax = self._colormap.vmax
        else: # bright areas
            new_vmax = min(self._colormap.vmax + 2, 12) # make sure vmax doesn't go above 12
            new_vmin = self._colormap.vmin

        return new_vmin, new_vmax

    def reset_colormap_hover(self):
        # reset vmin/vmax to original values when mouse hover hasn't triggered for a second
        if time.time() - self._last_hover_time > 1: # if hover hasn't triggered for a second
            self.vmin = self.original_vmin # set current vmin/vmax to original values
            self.vmax = self.original_vmax
            self.invalidate(DrawReason.CHANGE) # mark that the visualizer needs to be updated

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

        if value is not None:
            # see if we can get it. Assume it'll be cached, so this won't waste time.
            try:
                self.data_loader.get_named_quantity(value)
            except Exception as e:
                raise ValueError(f"Unable to get quantity named '{value}'") from e

        self.data_loader.quantity_name = value
        self.vmin_vmax_is_set = False
        self._reinitialize_colormap_and_bar()
        self.invalidate()

    def _reinitialize_colormap_and_bar(self):
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

    # Note: ChatGPT and Copilot used for debugging and explanations - tried to use get_sph_image to get color sample but get_sph_image can only be called once per frame
    # visualizer already has offscreen attribute
    # -> get offscreen texture (for calculating pixel intensity) in rg32float format
    # -> returns numpy array where img data can be extracted from (height, width, pixel coords/values)
    def get_offscreen_image(self) -> np.ndarray:
        texture = self.render_texture
        # because format "rg32float" uses 8 bytes per pixel
        bytes_per_pixel = 8  
        # calculate smallest multiple of 256 which can contain one row
        bytes_per_row = math.ceil(self._render_resolution * bytes_per_pixel / 256) * 256

        # from get_presentation_image, adapted for rg32float format
        data = self.device.queue.read_texture(
            {
                'texture': texture,
                'mip_level': 0,
                'origin': (0, 0, 0)
            },
            {
                'offset': 0,
                'bytes_per_row': bytes_per_row,
                'rows_per_image': self._render_resolution,
            },
            (self._render_resolution, self._render_resolution, 1)
        )

        # convert raw data to numpy array
        full_buffer = np.frombuffer(data, dtype=np.float32)
        
        # calculate floats each row has
        floats_per_pixel = 2 # rg32float has 2 floats per pixel
        floats_per_row_required = self._render_resolution * floats_per_pixel
        floats_per_row = bytes_per_row // 4  # 4 bytes per float

        # get pixel data from each row
        image = np.empty((self._render_resolution, self._render_resolution, floats_per_pixel), dtype=np.float32)
        for i in range(self._render_resolution):
            start = i * floats_per_row
            end = start + floats_per_row_required
            image[i] = full_buffer[start:end].reshape((self._render_resolution, floats_per_pixel))
        return image

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


class Visualizer(view_synchronizer.SynchronizationMixin, VisualizerBase):
    pass