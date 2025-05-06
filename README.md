https://github.com/user-attachments/assets/b185c3b8-8658-4f7d-96de-e976959e7ad6


topsy
=====

[![Build Status](https://github.com/pynbody/topsy/actions/workflows/build-test.yaml/badge.svg)](https://github.com/pynbody/topsy/actions)

This package visualises simulations, and is an add-on to the [pynbody](https://github.com/pynbody/pynbody) analysis package.
Its name nods to the [TIPSY](https://github.com/N-BodyShop/tipsy) project.
It is built using [wgpu](https://wgpu.rs), which is a future-facing GPU standard (with thanks to the [python wgpu bindings](https://wgpu-py.readthedocs.io/en/stable/guide.html)).

At the moment, `topsy` is a bit of a toy project, but it already works quite well with zoom 
(or low resolution) simulations. The future development path will depend on the level
of interest from the community.

UofM Spring 2025 Capstone
----------
Functionality implemented during UofM Spring 2025 Capstone by our team
- Contrast slider to shift colormap limits and boost dark/bright areas
  - Can be used by interacting with the 'Contrast' slider on the bottom toolbar
  - Pressing 'r' resets the slider to default
- Manually set colormap limits (vmin/vmax)
  - Can be used by pressing on the 'Set vmin/vmax' button on the bottom toolbar
- Import custom colormaps
  - Can be used by pressing on the 'Import Colormap' button on the bottom toolbar
  - The window displays a preview of the custom colormap
  - Use 'apply' to apply the previewed custom colormap
  - Use 'reset' to reset the custom colormap to the most recently selected matplotlib colormap
  - Created custom colormaps for testing with this online resource: https://jdherman.github.io/colormap/
- Split screen
  - Renders two of the 3D visualizations
  - Movements are synced with each other
  - Can be toggled by pressing the "Split" button on the toolbar
- iPython REPL
  - Do live debugging and coding with an iPython REPL integrated into Topsy
  - Allows for learning and experimenting with Topsy's modules.
  - When running Topsy, add the --repl flag: $ topsy file --repl
  - Developers can test and modify visualization components in real time.
- 3D Region Selection
  - Place a sphere within the visualization to get data about the particles within the sphere
  - Press the 'sphere' button on the toolbar to toggle
  - Move sphere with w,a,s,d,q,e keys
  - Resize sphere with \[ and \] keys
  - Show data properties within the sphere with f key

