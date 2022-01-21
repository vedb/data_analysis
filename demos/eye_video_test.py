"""
Created on Wed Nov 17 3:02:12 2021

Usage: python eye_video_test.py [None] [None]

        Parameters
        ----------

        None: tuple, len 2
            

        None: tuple, len 2
            

        None: str
            

@author: KamranBinaee
"""

import pupil_recording_interface as pri
import pandas as pd
import numpy as np
import os
import cv2
import sys
import logging

# %%
# Set Pupil Core generation
# -------------------------
#
# Set the generation of your Pupil Core device (1, 2 or 3)
pupil_gen = 2

# %%
# Set up stream configurations
# ----------------------------
configs = [
    pri.VideoStream.Config(
        device_type="uvc",
        device_uid=f"Pupil Cam{pupil_gen} ID0",
        name="eye0",
        resolution=(400, 400) if pupil_gen == 1 else (192, 192),
        fps=60,
        color_format="gray",
        pipeline=[pri.PupilDetector.Config(),
                pri.VideoDisplay.Config(flip=True, overlay_pupil=True)]
    ),
    pri.VideoStream.Config(
        device_type="uvc",
        device_uid=f"Pupil Cam{pupil_gen} ID1",
        name="eye1",
        resolution=(400, 400) if pupil_gen == 1 else (192, 192),
        fps=60,
        color_format="gray",
        pipeline=[pri.PupilDetector.Config(),
                pri.VideoDisplay.Config(overlay_pupil=True)],
    ),
]

# %%
# Set up logger
# -------------
logging.basicConfig(
    stream=sys.stdout, level=logging.INFO, format="%(message)s"
)

# %%
# Run manager
# -----------
#
# .. note::
#
#     When running the script from the command line, press 'Ctrl+C' to stop the
#     manager. When running from a Jupyter notebook, interrupt the kernel
#     (*Kernel > Interrupt Kernel* or press 'Esc' and then twice 'i').
updated_dict = {
                   "Absolute Exposure Time": 5000,
                   "Brightness": -20,
                   "Contrast": 10,
                   "Gamma": 100
}
with pri.StreamManager(configs) as manager:
    while not manager.stopped:
        if manager.all_streams_running:
            status = manager.format_status(
                "fps", format="{:.2f} Hz", max_cols=72, sleep=0.1
            )
            print("\r" + status, end="")
        if manager.keypresses._getvalue():
            key = manager.keypresses.popleft()
            if key.lower() == "t":
                # print(manager.streams["eye0"].device.capture)
                # print(type(manager.streams["eye0"].device.capture))
                # eye0_capture = manager.streams["eye0"].device.get_capture(f"Pupil Cam2 ID0", (400,400), 60)
                manager.streams["eye0"].device.controls(updated_dict)

print("\nStopped")
