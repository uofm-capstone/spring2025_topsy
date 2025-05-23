from __future__ import annotations

import logging
import numpy as np
import re
import wgpu
import matplotlib

from . import config

from .util import load_shader, preprocess_shader

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .visualizer import Visualizer


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



class Colormap:

    def __init__(self, visualizer: Visualizer, weighted_average: bool = False):
        self._visualizer = visualizer
        self._device = visualizer.device
        self._colormap_name = visualizer.colormap_name
        self._input_texture = visualizer.render_texture
        self._output_format = visualizer.canvas_format
        self._weighted_average = weighted_average

        self.vmin, self.vmax = 0,1
        self._log_scale = True
        # all three of these will be reset by set_vmin_vmax

        self._setup_texture()
        self._setup_shader_module()
        self._setup_render_pipeline()

    @property
    def log_scale(self):
        return self._log_scale

    @log_scale.setter
    def log_scale(self, value):
        old_value = self._log_scale
        self._log_scale = value
        if value != old_value:
            self._setup_shader_module()
            self._setup_render_pipeline()

    def _setup_shader_module(self):
        shader_code = load_shader("colormap.wgsl")
        # hack because at present we can't use const values in the shader to compile
        mode = "WEIGHTED_MEAN" if self._weighted_average else "DENSITY"
        active_flags = [mode]

        if self.log_scale:
            active_flags.append("LOG_SCALE")

        shader_code = preprocess_shader(shader_code, active_flags)

        self._shader = self._device.create_shader_module(code=shader_code, label="colormap")

    def set_custom_lut(self, lut: np.ndarray):
        assert lut.shape == (256, 3) # lut must be 256x3
        self.custom_lut = lut.astype(np.float32)
        self.use_custom_lut = True # flag to indicate that we are using a custom LUT
        self.invalidate_texture() # force a recompile of the shader

    # helper to update colorbar (which uses a different class than the colormap)
    def to_matplotlib(self):
        if hasattr(self, "custom_lut") and getattr(self, "use_custom_lut", False): # if its a custom lut and flag is not set
            return matplotlib.colors.ListedColormap(self.custom_lut, name="Custom") # return as a custom matplotlib colormap obj
        return matplotlib.colormaps[self._colormap_name] # return one of the matplotlib default colormaps

    def _setup_texture(self, num_points=config.COLORMAP_NUM_SAMPLES):
        if hasattr(self, "custom_lut") and getattr(self, "use_custom_lut", False): # if its a custom lut and flag is not set
            # create a texture from the custom LUT
            rgba = np.zeros((num_points, 4), dtype=np.float32)
            rgba[:, :3] = self.custom_lut
            rgba[:, 3] = 1.0
        else: # use the default colormap
            cmap_name = self._colormap_name if self._colormap_name in matplotlib.colormaps else "twilight_shifted"
            cmap = matplotlib.colormaps[cmap_name] # one of the matplotlib default colormaps
            rgba = cmap(np.linspace(0.001, 0.999, num_points)).astype(np.float32) # create a texture from the colormap

        self._texture = self._device.create_texture(
            label="colormap_texture",
            size=(num_points, 1, 1),
            dimension=wgpu.TextureDimension.d1,
            format=wgpu.TextureFormat.rgba32float,
            mip_level_count=1,
            sample_count=1,
            usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING
        )

        self._device.queue.write_texture(
            {
                "texture": self._texture,
                "mip_level": 0,
                "origin": [0, 0, 0],
            },
            rgba.tobytes(),
            {
                "bytes_per_row": 4 * 4 * num_points,
                "offset": 0,
            },
            (num_points, 1, 1)
        )
    
    # updates the gpu texture to use the custom colormap (if it exists)
    def invalidate_texture(self):
        # check if custom lut exists
        if not hasattr(self, "custom_lut") or not self.use_custom_lut:
            logger.warning("No custom LUT set. Skipping texture invalidation.")
            return
        
        # create rgba array from custom lut
        rgba = np.zeros((256, 4), dtype=np.float32)
        rgba[:, :3] = self.custom_lut  # use rgb from custom lut
        rgba[:, 3] = 1.0               # set alpha to 1

        # get size (256)
        num_points = rgba.shape[0]

        # create a texture from the custom lut
        self._texture = self._device.create_texture(
            label="custom_colormap_texture",
            size=(num_points, 1, 1),
            dimension=wgpu.TextureDimension.d1, # 1d texture
            format=wgpu.TextureFormat.rgba32float, # rgba32float format
            mip_level_count=1,
            sample_count=1,
            usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING
        )

        # write the texture to the GPU (transfers lut from cpu (numpy) to gpu (webgpu))
        self._device.queue.write_texture(
            {
                "texture": self._texture,
                "mip_level": 0,
                "origin": [0, 0, 0],
            },
            rgba.tobytes(), # converts numpy array to bytes
            {
                "bytes_per_row": 4 * 4 * num_points,
                "offset": 0,
            },
            (num_points, 1, 1)
        )

        # create a new bind group with the new texture because gpu bind groups are immutable
        # and we need to rebind the new texture to the bind group
        self._bind_group = self._device.create_bind_group(
            label="colormap_bind_group_custom",
            layout=self._bind_group_layout,
            entries=[
                {"binding": 0, "resource": self._input_texture.create_view()},
                {"binding": 1, "resource": self._input_interpolation},
                {"binding": 2, "resource": self._texture.create_view()},
                {"binding": 3, "resource": self._input_interpolation},
                {"binding": 4, "resource": {"buffer": self._parameter_buffer,
                                            "offset": 0,
                                            "size": self._parameter_buffer.size}}
            ]
        )

    def _setup_render_pipeline(self):
        self._parameter_buffer = self._device.create_buffer(size =4 * 3,
                                                            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)

        self._bind_group_layout = \
            self._device.create_bind_group_layout(
                label="colormap_bind_group_layout",
                entries=[
                    {
                        "binding": 0,
                        "visibility": wgpu.ShaderStage.FRAGMENT,
                        "texture": {"sample_type": wgpu.TextureSampleType.float,
                                    "view_dimension": wgpu.TextureViewDimension.d2},
                    },
                    {
                        "binding": 1,
                        "visibility": wgpu.ShaderStage.FRAGMENT,
                        "sampler": {"type": wgpu.SamplerBindingType.filtering},
                    },
                    {
                        "binding": 2,
                        "visibility": wgpu.ShaderStage.FRAGMENT,
                        "texture": {"sample_type": wgpu.TextureSampleType.float,
                                    "view_dimension": wgpu.TextureViewDimension.d1},
                    },
                    {
                        "binding": 3,
                        "visibility": wgpu.ShaderStage.FRAGMENT,
                        "sampler": {"type": wgpu.SamplerBindingType.filtering},
                    },
                    {
                        "binding": 4,
                        "visibility": wgpu.ShaderStage.FRAGMENT | wgpu.ShaderStage.VERTEX,
                        "buffer": {"type": wgpu.BufferBindingType.uniform}
                    }
                ]
            )

        self._input_interpolation = self._device.create_sampler(label="colormap_sampler",
                                                                mag_filter=wgpu.FilterMode.linear, )

        self._bind_group = \
            self._device.create_bind_group(
                label="colormap_bind_group",
                layout=self._bind_group_layout,
                entries=[
                    {"binding": 0,
                     "resource": self._input_texture.create_view(),
                     },
                    {"binding": 1,
                     "resource": self._input_interpolation,
                     },
                    {"binding": 2,
                     "resource": self._texture.create_view(),
                     },
                    {"binding": 3,
                     "resource": self._input_interpolation,
                     },
                    {"binding": 4,
                        "resource": {"buffer": self._parameter_buffer,
                                     "offset": 0,
                                     "size": self._parameter_buffer.size}
                     }
                ]
            )

        self._pipeline_layout = \
            self._device.create_pipeline_layout(
                label="colormap_pipeline_layout",
                bind_group_layouts=[self._bind_group_layout]
            )


        self._pipeline = \
            self._device.create_render_pipeline(
                layout=self._pipeline_layout,
                label="colormap_pipeline",
                vertex={
                    "module": self._shader,
                    "entry_point": "vertex_main",
                    "buffers": []
                },
                primitive={
                    "topology": wgpu.PrimitiveTopology.triangle_strip,
                },
                depth_stencil=None,
                multisample=None,
                fragment={
                    "module": self._shader,
                    "entry_point": "fragment_main",
                    "targets": [
                        {
                            "format": self._output_format,
                            "blend": {
                                "color": {
                                    "src_factor": wgpu.BlendFactor.one,
                                    "dst_factor": wgpu.BlendFactor.zero,
                                    "operation": wgpu.BlendOperation.add,
                                },
                                "alpha": {
                                    "src_factor": wgpu.BlendFactor.one,
                                    "dst_factor": wgpu.BlendFactor.zero,
                                    "operation": wgpu.BlendOperation.add,
                                },
                            }
                        }
                    ]
                }
            )

    def encode_render_pass(self, command_encoder, target_texture_view):
        self._update_parameter_buffer(target_texture_view.size[0], target_texture_view.size[1])
        colormap_render_pass = command_encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": target_texture_view,
                    "resolve_target": None,
                    "clear_value": (0.0, 0.0, 0.0, 1.0),
                    "load_op": wgpu.LoadOp.load,
                    "store_op": wgpu.StoreOp.store,
                }
            ]
        )
        colormap_render_pass.set_pipeline(self._pipeline)
        colormap_render_pass.set_bind_group(0, self._bind_group, [], 0, 99)
        colormap_render_pass.draw(4, 1, 0, 0)
        colormap_render_pass.end()


    def autorange_vmin_vmax(self):
        """Set the vmin and vmax values for the colomap based on the most recent SPH render"""

        # This can and probably should be done on-GPU using a compute shader, but for now
        # we'll do it on the CPU
        vals = self._visualizer.get_sph_image().ravel()

        if (vals<0).any():
            self.log_scale = False
        else:
            self.log_scale = True
        # NB above switching of log scale will automatically rebuild the pipeline if needed

        if self.log_scale:
            vals = np.log10(vals)

        vals = vals[np.isfinite(vals)]
        if len(vals) > 200:
            self.vmin, self.vmax = np.percentile(vals, [1.0, 99.9])
        elif len(vals)>2:
            self.vmin, self.vmax = np.min(vals), np.max(vals)
        else:
            logger.warning(
                "Problem setting vmin/vmax, perhaps there are no particles or something is wrong with them?")
            logger.warning("Press 'r' in the window to try again")
            self.vmin, self.vmax = 0.0, 1.0


    def _update_parameter_buffer(self, width, height):
        parameter_dtype = [("vmin", np.float32, (1,)),
                           ("vmax", np.float32, (1,)),
                           ("window_aspect_ratio", np.float32, (1,))]

        parameters = np.zeros((), dtype=parameter_dtype)
        parameters["vmin"] = self.vmin
        parameters["vmax"] = self.vmax


        parameters["window_aspect_ratio"] = float(width)/height
        self._device.queue.write_buffer(self._parameter_buffer, 0, parameters)