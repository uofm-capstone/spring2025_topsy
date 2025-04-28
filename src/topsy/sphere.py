from __future__ import annotations

import numpy as np
import wgpu
from .line import Line

class SphereOverlay(Line):
    def __init__(self, visualizer, position=(0, 0, 0), radius=1.0, segments=32, color=(1, 1, 0, 1), width=2.0):
        self._position = np.array(position, dtype=np.float32)
        self._radius = radius
        self._segments = segments

        # Generate line path
        path = self._generate_wireframe_sphere()

        super().__init__(visualizer, path, color, width)

    def _generate_wireframe_sphere(self):
        """
        Generate latitudinal and longitudinal lines forming a wireframe sphere.
        """
        lines = []
        pos = self._position
        r = self._radius
        seg = self._segments

        # Latitude rings (XY plane)
        for i in range(1, seg):
            lat = np.pi * i / seg
            z = r * np.cos(lat)
            ring_radius = r * np.sin(lat)
            for j in range(seg):
                theta1 = 2 * np.pi * j / seg
                theta2 = 2 * np.pi * (j + 1) / seg
                p1 = pos + ring_radius * np.array([np.cos(theta1), np.sin(theta1), 0])
                p2 = pos + ring_radius * np.array([np.cos(theta2), np.sin(theta2), 0])
                lines.append((*p1, 1.0))
                lines.append((*p2, 1.0))

        # Longitude rings (XZ and YZ planes)
        for j in range(seg):
            theta = 2 * np.pi * j / seg
            for i in range(seg):
                phi1 = np.pi * i / seg
                phi2 = np.pi * (i + 1) / seg
                p1 = pos + r * np.array([
                    np.sin(phi1) * np.cos(theta),
                    np.sin(phi1) * np.sin(theta),
                    np.cos(phi1)
                ])
                p2 = pos + r * np.array([
                    np.sin(phi2) * np.cos(theta),
                    np.sin(phi2) * np.sin(theta),
                    np.cos(phi2)
                ])
                lines.append((*p1, 1.0))
                lines.append((*p2, 1.0))

        return np.array(lines, dtype=np.float32)

    def set_position_and_radius(self, position, radius):
        self._position = np.array(position, dtype=np.float32)
        self._radius = radius
        self.path = self._generate_wireframe_sphere()  # triggers buffer rebuild    
        self._setup_buffers()

    def set_radius(self, radius):
        self._radius = radius
        self.path = self._generate_wireframe_sphere()

        # Rebuild line segments
        self._line_starts = np.ascontiguousarray(self.path[::2, :])
        self._line_ends = np.ascontiguousarray(self.path[1::2, :])

        # Upload to GPU
        self._device.queue.write_buffer(self._vertex_buffer_starts, 0, self._line_starts)
        self._device.queue.write_buffer(self._vertex_buffer_ends, 0, self._line_ends)

        # Redraw
        self._visualizer.invalidate()


    @property
    def position(self):
        return self._position
    
    # function taken from existing simcube.py to render the sphere in the correct position and as a 3d object
    def encode_render_pass(self, command_encoder: wgpu.GPUCommandEncoder,
                                target_texture_view: wgpu.GPUTextureView):
        self._params["transform"] = (
            self._visualizer._sph.last_transform_params["transform"]
            @ self._visualizer.sph_clipspace_to_screen_clipspace_matrix()
        )
        super().encode_render_pass(command_encoder, target_texture_view)


