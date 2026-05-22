import asyncio
import base64
import copy
import os
import typing
from collections import OrderedDict
from io import BytesIO
from pathlib import Path

import cv2
import flet as ft
import numpy as np
import tifffile
from PIL import Image, ImageEnhance

from drawing_tool import DrawingTool
from drawing_util import search_free_id, mask_shifting, rgb_to_hex


def load_image(image, auto_adjust=False, get_slice=-1, brightness=1.0, contrast=1.0):
    shape = list(image.shape)

    if image.dtype == np.uint16:
        #16bit case
        max_val = 65535
        cv_target_dtype = cv2.CV_16U
    else:
        #8bit case
        max_val = 255
        cv_target_dtype = cv2.CV_8U

    check = image.ndim == 3
    if check:
        if not get_slice == -1:
            image = image[get_slice, :, :]
        else:
            image = np.max(image, axis=0)

    if auto_adjust:
        image = cv2.normalize(image, None, alpha=0, beta=max_val, norm_type=cv2.NORM_MINMAX,dtype=cv_target_dtype)
    elif brightness != 1.0 or contrast != 1.0:
        mean_lum = np.mean(image)

        mid_val = mean_lum * brightness

        alpha = brightness * contrast
        beta = mid_val * (1 - contrast)

        image = cv2.addWeighted(image, alpha, image, 0, beta, dtype=cv_target_dtype)

    if image.dtype == np.uint16:
        image = cv2.convertScaleAbs(image, alpha=1 / 256.0)

    _, buffer = cv2.imencode('.png', image, [cv2.IMWRITE_PNG_COMPRESSION, 1])

    return base64.b64encode(buffer).decode('utf-8'), shape, check


def convert_npy_to_canvas(mask, outline, mask_color, outline_color, opacity, slice_id=-1):
    """
    handles the conversion of the given file data

    Args:
        mask= the mask data stored in the numpy directory
        outline= the outline data stored in the numpy directory
    """
    if mask.ndim == 3:  # if 3d get the given slice or get a max projection
        mask_slice = mask[slice_id] if slice_id >= 0 else mask.any(axis=0)
        outline_slice = outline[slice_id] if slice_id >= 0 else outline.any(axis=0)
    else:  # 2d nothing to do
        mask_slice = mask
        outline_slice = outline

    # filter values greater than 0
    mask_bool = mask_slice > 0
    outline_bool = outline_slice > 0

    has_mask = mask_bool.any()
    has_outline = outline_bool.any()

    image_mask = np.zeros((mask_slice.shape[0], mask_slice.shape[1], 4),
                          dtype=np.uint8)  # uint8 because here we dont use the cell_id's

    if has_mask:
        image_mask[mask_bool] = (mask_color[2], mask_color[1], mask_color[0], opacity)

    if has_outline:
        image_mask[outline_bool] = (outline_color[2], outline_color[1], outline_color[0], 255)

    encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]
    success, buffer = cv2.imencode('.png', image_mask, encode_params)

    return base64.b64encode(buffer).decode('utf-8')


def _get_cell_id_from_position(position, mask):
    """
    Get the cell ID from the clicked position.
    """
    x, y = int(position[0]), int(position[1])
    if mask.ndim == 3:
        if 0 <= y < mask.shape[1] and 0 <= x < mask.shape[2]:
            return mask[:,y, x]
        return None
    else:
        if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
            return mask[y, x]
        return None


class FluorescenceCache:
    def __init__(self, max_values=20):
        self.fluorescence_cache = OrderedDict()
        self._max_values = max_values

    def clear(self):
        self.fluorescence_cache.clear()

    def get_fluorescence_value(self, cell_id, mask, np_image, image_dim, channel, zslice=None):
        if zslice == -1:
            zslice = None
        if image_dim not in self.fluorescence_cache:
            self.fluorescence_cache[image_dim] = OrderedDict()
        if channel not in self.fluorescence_cache[image_dim]:
            self.fluorescence_cache[image_dim][channel] = OrderedDict()
        if zslice not in self.fluorescence_cache[image_dim][channel]:
            self.fluorescence_cache[image_dim][channel][zslice] = OrderedDict()

        if cell_id in self.fluorescence_cache[image_dim][channel][zslice]:
            self.fluorescence_cache[image_dim][channel][zslice].move_to_end(cell_id)
            self.fluorescence_cache[image_dim][channel].move_to_end(zslice)
            self.fluorescence_cache[image_dim].move_to_end(channel)
            self.fluorescence_cache.move_to_end(image_dim)
            return self.fluorescence_cache[image_dim][channel][zslice][cell_id]

        if len(self.fluorescence_cache[image_dim][channel][zslice]) > self._max_values:
            self.fluorescence_cache[image_dim][channel][zslice].popitem(last=False)

        cell_mask = mask == cell_id
        val = float(np.mean(np_image[cell_mask]))

        self.fluorescence_cache[image_dim][channel][zslice][cell_id] = val
        self.fluorescence_cache[image_dim][channel][zslice].move_to_end(cell_id)

        return val


class ImageCache:
    def __init__(self, max_images=5):
        self.cache = OrderedDict()
        self._max_images = max_images

    def get_image(self, path):
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        else:
            image = tifffile.imread(path)
            self.add_image(path, image)
            return image

    def add_image(self, path, data):
        self.cache[path] = data
        self.cache.move_to_end(path)
        if len(self.cache) > self._max_images:
            self.cache.popitem(last=False)

    def clear(self):
        self.cache.clear()


class ImageEditingView(ft.Card):
    def __init__(self, on_mask_change: typing.Callable[[str, bool], None] = None):
        super().__init__()
        self._mask_paths = None
        self._main_paths = None

        self._mask_path = None  # Could set a mask_path for TESTING
        self._mask_data = None  # np.load(Path(self._mask_path), allow_pickle=True).item()
        self._slice_id = -1
        self._image_3d = False
        self._image_id = None
        self._channel_id = None
        self._seg_channel_id = None
        self._save_lock = None
        self._image_cache = ImageCache()
        self._fluorescence_cache = FluorescenceCache()
        self._running_tasks = set()
        self.brightness = 1.0
        self.contrast = 1.0
        self.auto_adjust = False
        self.mask_color = (255, 0, 0)
        self.outline_color = (0, 255, 0)
        self.mask_opacity = 128
        self._user_2_5d = False
        self.on_mask_change = on_mask_change or (lambda y, x: None)
        self.mask_suffix = "_seg"
        self.expand = True
        self._redo_stack = []
        self._undo_stack = []
        self._edit_allowed = True
        self._mask_image = ft.Image(src="Placeholder", fit=ft.BoxFit.CONTAIN, visible=False, gapless_playback=True,
                                    expand=True,left=0, right=0, top=0, bottom=0)
        self._main_image = ft.Image(src="Placeholder", fit=ft.BoxFit.CONTAIN, visible=False, gapless_playback=True,
                                    expand=True,left=0, right=0, top=0, bottom=0)
        self.drawing_tool = DrawingTool(on_cell_drawn=self._cell_drawn, on_cell_deleted=self._delete_cell,
                                        on_show_ids=self._handle_show_ids)

        self._mask_button = ft.IconButton(icon=ft.Icons.REMOVE_RED_EYE, icon_color=ft.Colors.BLACK12,
                                          style=ft.ButtonStyle(
                                              shape=ft.RoundedRectangleBorder(radius=12), ),
                                          on_click=self._show_mask,
                                          tooltip="Show Mask", hover_color=ft.Colors.WHITE12, disabled=True)
        self._edit_button = ft.IconButton(icon=ft.Icons.BRUSH, icon_color=ft.Colors.BLACK_12,
                                          style=ft.ButtonStyle(
                                              shape=ft.RoundedRectangleBorder(radius=12), ), disabled=True,
                                          tooltip="Draw mode", hover_color=ft.Colors.WHITE_12,
                                          on_click=self._toggle_draw)
        self._delete_button = ft.IconButton(icon=ft.Icons.CLEAR, icon_color=ft.Colors.BLACK_12,
                                            style=ft.ButtonStyle(
                                                shape=ft.RoundedRectangleBorder(radius=12), ), disabled=True,
                                            tooltip="Delete mode", hover_color=ft.Colors.WHITE12,
                                            on_click=self._toggle_delete)
        self._delete_mask_button = ft.IconButton(icon=ft.Icons.DELETE_FOREVER, icon_color=ft.Colors.WHITE_60,
                                                 style=ft.ButtonStyle(
                                                     shape=ft.RoundedRectangleBorder(radius=12), ),
                                                 tooltip="Delete the complete mask", hover_color=ft.Colors.WHITE12,
                                                 on_click=lambda e: self.delete_mask())
        self._redo_button = ft.IconButton(icon=ft.Icons.REDO_SHARP, icon_color=ft.Colors.BLACK_12,
                                          style=ft.ButtonStyle(
                                              shape=ft.RoundedRectangleBorder(radius=12), ),
                                          tooltip="Redo action", hover_color=ft.Colors.WHITE_12,
                                          on_click=self.redo_stack, disabled=True)

        self._undo_button = ft.IconButton(icon=ft.Icons.UNDO_SHARP, icon_color=ft.Colors.BLACK_12,
                                          style=ft.ButtonStyle(
                                              shape=ft.RoundedRectangleBorder(radius=12), ),
                                          tooltip="Undo action", hover_color=ft.Colors.WHITE12,
                                          on_click=self.undo_stack, disabled=True)

        # controls for visible cell id and value, when hovered over the cell mask
        self._show_id_checkbox = ft.IconButton(
            icon=ft.CupertinoIcons.NUMBER_CIRCLE_FILL,
            icon_color=ft.Colors.BLACK_12,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=12), ),
            hover_color=ft.Colors.WHITE12,
            selected=False,
            disabled=True,
            on_click=self._toggle_cell_info,
            tooltip="By hovering over the image: Show ids and values of the cells."
        )
        self._id_info = ft.Container(
            content=ft.DataTable(
                columns=[
                    ft.DataColumn(label=ft.Text("ID", color=ft.Colors.WHITE)),
                    ft.DataColumn(label=ft.Text("Value", color=ft.Colors.WHITE)),
                ],
                rows=[
                ],
                border=ft.Border.all(1, ft.Colors.OUTLINE_VARIANT),
                border_radius=10,
                bgcolor=ft.Colors.BLACK_26,
                width=120,
                column_spacing=4,

            ),
            border_radius=10,
            visible=False,
            right=5,
            top=5,
            ignore_interactions=True,
        )


        self._slider_2_5d = ft.Slider(
            min=0, max=100, divisions=None, label="Slice: {value}", value=0,
            opacity=1.0 if self._user_2_5d else 0.0, height=20, width=170,
            active_color=ft.Colors.WHITE60, thumb_color=ft.Colors.WHITE, disabled=True,
            animate_opacity=ft.Animation(duration=600, curve=ft.AnimationCurve.LINEAR_TO_EASE_OUT),
            on_change=self._slider2_5d_change
        )
        self._slider_2d = ft.CupertinoSlidingSegmentedButton(
            selected_index=0 if not self._user_2_5d else 1,
            thumb_color=ft.Colors.WHITE,
            bgcolor=ft.Colors.WHITE60,
            padding=ft.Padding.symmetric(vertical=0,horizontal=0),
            controls=[
                ft.Text("2D", color=ft.Colors.BLACK, weight=ft.FontWeight.BOLD),
                ft.Text("2.5D", color=ft.Colors.BLACK, weight=ft.FontWeight.BOLD)
            ],
            on_change=self._slider2d_update
        )
        self._shifting_check_box = ft.IconButton(
            icon=ft.Icon(ft.Icons.FORMAT_LIST_NUMBERED, color=ft.Colors.WHITE60),
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=12), ),
            hover_color=ft.Colors.WHITE12,
            selected_icon=ft.Icon(ft.Icons.FORMAT_LIST_NUMBERED, color=ft.Colors.WHITE),
            selected=False,
            on_click=self._toggle_shifting,
            tooltip="Shifting IDs: OFF \nDeleted masks will leave gaps in the ID sequence. No shifting will occur."
        )
        self.control_tools = ft.Container(ft.Container(ft.Row(
            [self._undo_button,
             self._redo_button,
             self._shifting_check_box,
             self._edit_button,
             self._delete_button,
             self._mask_button,
             self._slider_2d,
             ft.Container(
                 content=self._slider_2_5d,
                 theme=ft.Theme(
                     slider_theme=ft.SliderTheme(
                         value_indicator_text_style=ft.TextStyle(color=ft.Colors.BLACK, size=15,
                                                                 weight=ft.FontWeight.BOLD),
                     )
                 ),
                 dark_theme=ft.Theme(
                     slider_theme=ft.SliderTheme(
                         value_indicator_text_style=ft.TextStyle(color=ft.Colors.BLACK, size=15,
                                                                 weight=ft.FontWeight.BOLD),
                     )
                 ),
             ),
             self._delete_mask_button,
             self._show_id_checkbox,
             ], spacing=2, alignment=ft.MainAxisAlignment.CENTER, height=38,
        ), bgcolor=ft.Colors.BLUE_ACCENT, expand=True, border_radius=ft.border_radius.vertical(top=0, bottom=12),

        ))
        self.image_stack = ft.InteractiveViewer(content=ft.Stack([self._main_image,
                                                                  self._mask_image,
                                                                  self.drawing_tool,
                                                                  ], expand=True), expand=True)

        self.content = ft.Stack([
            ft.Column(controls=[ft.Container(self.image_stack, alignment=ft.Alignment.CENTER, expand=True),
                                self.control_tools], spacing=0),
            self._id_info,
        ])

    def set_mask_paths(self, mask_paths: list):
        self._mask_paths = mask_paths

    def set_main_paths(self, main_paths: list):
        self._main_paths = main_paths

    def set_colors(self, mask_color, outline_color, opacity):
        if mask_color is not None:
            self.mask_color = mask_color
        if outline_color is not None:
            self.drawing_tool.draw_color = rgb_to_hex(outline_color)
            self.outline_color = outline_color
        if opacity is not None:
            self.mask_opacity = opacity
        self.page.run_task(self.update_mask_image)

    def reset_image(self, without_update=False):
        self._main_image.src = "Placeholder"
        self._main_image.visible = False
        self._seg_channel_id = None
        self._image_id = None
        self._mask_path = None
        self._mask_data = None
        self._mask_image.src = "Placeholder"
        self._mask_image.visible = False
        self._mask_button.tooltip = "Show mask"
        self._mask_button.icon_color = ft.Colors.BLACK12
        self._mask_button.disabled = True
        self._edit_button.icon_color = ft.Colors.BLACK12
        self._edit_button.disabled = True
        self._delete_button.icon_color = ft.Colors.BLACK_12
        self._delete_button.disabled = True
        self.drawing_tool.deactivate_drawing()
        self.drawing_tool.deactivate_delete()
        self._redo_stack.clear()
        self._undo_stack.clear()
        self._redo_button.disabled = True
        self._undo_button.disabled = True
        self._redo_button.icon_color = ft.Colors.BLACK_12
        self._redo_button.disabled = True
        self._undo_button.icon_color = ft.Colors.BLACK_12
        self._undo_button.disabled = True
        self._image_cache.clear()
        self._show_id_checkbox.disabled = True
        self._show_id_checkbox.icon_color = ft.Colors.BLACK_12
        self._show_id_checkbox.selected = False
        self.drawing_tool.deactivate_cell_info()
        self._id_info.visible = False
        self._fluorescence_cache.clear()
        self.cancel_all_tasks()
        self._edit_allowed = True
        self._delete_mask_button.icon_color = ft.Colors.WHITE_60
        self._delete_mask_button.disabled = False
        if not without_update:
            self._main_image.update()
            self._mask_image.update()
            self._mask_button.update()
            self._edit_button.update()
            self._delete_button.update()
            self._redo_button.update()
            self._undo_button.update()
            self._show_id_checkbox.update()
            self._id_info.update()
            self._delete_mask_button.update()

    def select_image(self, img_id, channel_id, seg_channel_id):
        self.page.run_task(self.select_image_async, img_id, channel_id, seg_channel_id)

    async def select_image_async(self, img_id, channel_id, seg_channel_id):
        if self._seg_channel_id != seg_channel_id or self._image_id != img_id:
            await self._load_mask_image(img_id, seg_channel_id)
            # reset undo/redo when a new image is selected
            if not self._mask_button.disabled:
                self._show_id_checkbox.disabled = False
                if self._show_id_checkbox.selected:
                    self._show_id_checkbox.icon_color = ft.Colors.WHITE
                else:
                    self._show_id_checkbox.icon_color = ft.Colors.WHITE_60
                self._show_id_checkbox.update()
            else:
                self._show_id_checkbox.disabled = True
                self._show_id_checkbox.selected = False
                self.drawing_tool.deactivate_cell_info()
                self._show_id_checkbox.icon_color = ft.Colors.BLACK_12
                self._show_id_checkbox.update()
                self._id_info.visible = False
                self._id_info.update()
            self._redo_stack.clear()
            self._undo_stack.clear()
            self._redo_button.disabled = True
            self._undo_button.disabled = True
            self._redo_button.icon_color = ft.Colors.BLACK_12
            self._redo_button.disabled = True
            self._undo_button.icon_color = ft.Colors.BLACK_12
            self._undo_button.disabled = True
            self._redo_button.update()
            self._undo_button.update()

        self._image_id = img_id
        self._channel_id = channel_id
        self._seg_channel_id = seg_channel_id
        self._load_main_image(img_id, channel_id)

    def _load_main_image(self, img_id, channel_id):
        if self._main_paths is not None:
            if img_id in self._main_paths:
                if channel_id in self._main_paths[img_id]:
                    self._load_main_image_with_path(self._main_paths[img_id][channel_id])
                    return
        self._image_3d = False
        self._main_image.src = "Placeholder"
        self._mask_image.visible = False
        self._main_image.update()

    async def _slider2d_update(self, e):
        if int(e.data) == 1:
            self._slider_2_5d.opacity = 1.0
            self._user_2_5d = True
        else:
            self._slider_2_5d.opacity = 0.0
            self._user_2_5d = False

        await self._slider2_5d_change()
        self._slider_2_5d.update()

    async def _slider2_5d_change(self, e=None):
        if self._user_2_5d:
            self._slice_id = int(self._slider_2_5d.value)
        else:
            self._slice_id = -1

        if self._main_image.src != "Placeholder":
            self._load_main_image(self._image_id, self._channel_id)
            self.page.run_task(self.update_mask_image)
            if self._image_3d:
                self._redo_stack.clear()
                self._undo_stack.clear()
                self._redo_button.disabled = True
                self._undo_button.disabled = True
                self._redo_button.icon_color = ft.Colors.BLACK_12
                self._undo_button.icon_color = ft.Colors.BLACK_12
                self._redo_button.update()
                self._undo_button.update()

    def cancel_all_tasks(self):
        for task in self._running_tasks:
            task.cancel()
        self._running_tasks.clear()

    async def _adjust_image_async(self, path, brightness, contrast):
        return await asyncio.to_thread(load_image, self._image_cache.get_image(path), False, self._slice_id, brightness,
                                       contrast)

    async def _update_main_image(self, path):
        """
        Updates the main image as base64_image with the new brightness and contrast values.
        """
        src, shape, img_3d = await self._adjust_image_async(path,
                                                            self.brightness,
                                                            self.contrast
                                                            )
        self._main_image.src = src
        self._main_image.update()

    async def update_main_image_with_brightness_contrast(self, path):
        task = asyncio.create_task(self._update_main_image(path))
        self._running_tasks.add(task)
        try:
            await task
        except asyncio.CancelledError:
            return
        finally:
            self._running_tasks.discard(task)

    def _load_main_image_with_path(self, path):
        self.cancel_all_tasks()
        src, shape, img_3d = load_image(self._image_cache.get_image(path), auto_adjust=self.auto_adjust,
                                        get_slice=self._slice_id, brightness=self.brightness, contrast=self.contrast)
        self._main_image.src = src
        self._main_image.visible = True
        self._main_image.update()
        if img_3d:
            self.drawing_tool.set_bounds(shape[2], shape[1])
            self._image_3d = True
            if self._slider_2_5d.opacity == 1.0 and self.check_edit_allowed():
                if self._edit_button.disabled:
                    self._edit_button.icon_color = ft.Colors.WHITE60
                    self._edit_button.disabled = False
                    self._edit_button.update()
                if self._delete_button.disabled:
                    self._delete_button.icon_color = ft.Colors.WHITE60
                    self._delete_button.disabled = False
                    self._delete_button.update()
                if not self._mask_button.disabled:
                    self._show_id_checkbox.disabled = False
                    if self._show_id_checkbox.selected:
                        self._show_id_checkbox.icon_color = ft.Colors.WHITE
                    else:
                        self._show_id_checkbox.icon_color = ft.Colors.WHITE_60
                    self._show_id_checkbox.update()
                else:
                    self._show_id_checkbox.disabled = True
                    self._show_id_checkbox.selected = False
                    self.drawing_tool.deactivate_cell_info()
                    self._show_id_checkbox.icon_color = ft.Colors.BLACK_12
                    self._show_id_checkbox.update()
                    self._id_info.visible = False
                    self._id_info.update()

            else:
                if self.check_edit_allowed():
                    self._edit_button.icon_color = ft.Colors.WHITE_60
                    self._edit_button.disabled = False
                    self._edit_button.update()
                    self._delete_button.icon_color = ft.Colors.WHITE_60
                    self._delete_button.disabled = False
                    self._delete_button.update()
                else:
                    self._edit_button.icon_color = ft.Colors.BLACK_12
                    self._edit_button.disabled = True
                    self._edit_button.update()
                    self.drawing_tool.deactivate_drawing()
                    self._delete_button.icon_color = ft.Colors.BLACK_12
                    self._delete_button.disabled = True
                    self._delete_button.update()
                    self.drawing_tool.deactivate_delete()
                if self._slider_2_5d.opacity != 1.0:
                    if not self._mask_button.disabled:
                        self._show_id_checkbox.disabled = False
                        if self._show_id_checkbox.selected:
                            self._show_id_checkbox.icon_color = ft.Colors.WHITE
                        else:
                            self._show_id_checkbox.icon_color = ft.Colors.WHITE_60
                        self._show_id_checkbox.update()
                    self._id_info.visible = False
                    self._id_info.update()
                    self._show_id_checkbox.update()
            self._slider_2_5d.value = 0 if shape[0] - 1 < self._slider_2_5d.value else self._slider_2_5d.value
            self._slider_2_5d.max = shape[0] - 1
            self._slider_2_5d.divisions = shape[0] - 1
            self._slider_2_5d.disabled = False
            self._slider_2_5d.update()
        else:
            self.drawing_tool.set_bounds(shape[1], shape[0])
            self._image_3d = False
            if self.check_edit_allowed():
                if self._edit_button.disabled:
                    self._edit_button.icon_color = ft.Colors.WHITE60
                    self._edit_button.disabled = False
                    self._edit_button.update()
                if self._delete_button.disabled:
                    self._delete_button.icon_color = ft.Colors.WHITE60
                    self._delete_button.disabled = False
                    self._delete_button.update()
            else:
                self._edit_button.icon_color = ft.Colors.BLACK_12
                self._edit_button.disabled = True
                self._edit_button.update()
                self.drawing_tool.deactivate_drawing()
                self._delete_button.icon_color = ft.Colors.BLACK_12
                self._delete_button.disabled = True
                self._delete_button.update()
                self.drawing_tool.deactivate_delete()
            self._slider_2_5d.value = 0
            self._slice_id = 0
            self._slider_2_5d.max = 1
            self._slider_2_5d.divisions = None
            self._slider_2_5d.disabled = True
            self._slider_2_5d.update()
        return

    async def _load_mask_image(self, img_id, seg_channel_id):
        if self._mask_paths is not None:
            if img_id in self._mask_paths:
                if seg_channel_id in self._mask_paths[img_id]:
                    new_path = self._mask_paths[img_id][seg_channel_id]
                    if new_path != self._mask_path:
                        self._mask_data = np.load(
                            Path(self._mask_paths[img_id][seg_channel_id]), allow_pickle=True).item()
                        self._mask_path = new_path
                        self._mask_data["masks"] = self._mask_data["masks"].astype(np.uint16)
                        self._mask_data["outlines"] = self._mask_data["outlines"].astype(np.uint16)

                    self._mask_image.src = await asyncio.to_thread(convert_npy_to_canvas, self._mask_data["masks"],
                                                                   self._mask_data["outlines"],
                                                                   self.mask_color, self.outline_color,
                                                                   self.mask_opacity,
                                                                   slice_id=self._slice_id)
                    self._mask_image.update()
                    if not self._mask_image.visible:
                        self._mask_button.icon_color = ft.Colors.WHITE60
                        self._mask_button.tooltip = "Show mask"
                        self._mask_button.disabled = False
                        self._mask_button.update()
                    return

        self._mask_path = None
        self._mask_data = None
        self._mask_image.src = "Placeholder"
        self._mask_image.visible = False
        self._mask_image.update()
        self._mask_button.tooltip = "Show mask"
        self._mask_button.icon_color = ft.Colors.BLACK12
        self._mask_button.disabled = True
        self._mask_button.update()

    async def update_mask_image(self, reset=False):
        if reset:
            self._mask_path = None
        if self._mask_path is not None:
            await self._async_update_mask_image()
        elif self._mask_paths is not None and self._image_id in self._mask_paths and self._seg_channel_id in \
                self._mask_paths[self._image_id] and self._mask_paths[self._image_id][self._seg_channel_id] is not None:
            self._mask_path = self._mask_paths[self._image_id][self._seg_channel_id]
            self._mask_data = np.load(Path(self._mask_path), allow_pickle=True).item()
            self._mask_data["masks"] = self._mask_data["masks"].astype(np.uint16)
            self._mask_data["outlines"] = self._mask_data["outlines"].astype(np.uint16)
            await self._async_update_mask_image()
        else:
            self._mask_image.src = "Placeholder"
            self._mask_image.visible = False
            self._mask_image.update()
            self._mask_button.tooltip = "Show mask"
            self._mask_button.icon_color = ft.Colors.BLACK12
            self._mask_button.disabled = True
            self._mask_button.update()

        if not self._mask_button.disabled:
            self._show_id_checkbox.disabled = False
            if self._show_id_checkbox.selected:
                self._show_id_checkbox.icon_color = ft.Colors.WHITE
            else:
                self._show_id_checkbox.icon_color = ft.Colors.WHITE_60
            self._show_id_checkbox.update()
        else:
            self._show_id_checkbox.disabled = True
            self._show_id_checkbox.selected = False
            self.drawing_tool.deactivate_cell_info()
            self._show_id_checkbox.icon_color = ft.Colors.BLACK_12
            self._show_id_checkbox.update()
            self._id_info.visible = False
            self._id_info.update()

    async def _async_update_mask_image(self):
        if self._mask_data is None:
            return
        mask = self._mask_data["masks"]
        outline = self._mask_data["outlines"]
        self._mask_image.src = await asyncio.to_thread(convert_npy_to_canvas, mask, outline, self.mask_color,
                                                       self.outline_color, self.mask_opacity, self._slice_id)
        self._mask_image.update()
        if not self._mask_image.visible:
            self._mask_button.icon_color = ft.Colors.WHITE60
            self._mask_button.tooltip = "Show mask"
            self._mask_button.disabled = False
            self._mask_button.update()

    async def _show_mask(self, e):
        self._mask_image.visible = not self._mask_image.visible
        self._mask_image.update()
        self._mask_button.icon_color = ft.Colors.WHITE if self._mask_image.visible else ft.Colors.WHITE60
        self._mask_button.tooltip = "Hide mask" if self._mask_image.visible else "Show mask"
        self._mask_button.update()

    async def _toggle_draw(self, e):
        self._edit_button.icon_color = ft.Colors.WHITE if self._edit_button.icon_color == ft.Colors.WHITE_60 else ft.Colors.WHITE60
        self._edit_button.update()
        if self._edit_button.icon_color == ft.Colors.WHITE:
            self._delete_button.icon_color = ft.Colors.WHITE60
            self._delete_button.update()
            self.drawing_tool.draw()
        else:
            self.drawing_tool.deactivate_drawing()

    async def _toggle_delete(self, e):
        self._delete_button.icon_color = ft.Colors.WHITE if self._delete_button.icon_color == ft.Colors.WHITE_60 else ft.Colors.WHITE60
        self._delete_button.update()
        if self._delete_button.icon_color == ft.Colors.WHITE:
            if not self._edit_button.disabled:
                self._edit_button.icon_color = ft.Colors.WHITE60
                self._edit_button.update()
                self.drawing_tool.delete()
        else:
            self.drawing_tool.deactivate_delete()

    async def _toggle_shifting(self, e):
        e.control.selected = not e.control.selected
        if e.control.selected:
            e.control.tooltip = "Shifting IDs: ON \nShifts the IDs when a mask is deleted to restore a continuous order without gaps."
        else:
            e.control.tooltip = "Shifting IDs: OFF \nDeleted masks will leave gaps in the ID sequence. No shifting will occur."

        e.control.update()

    async def _toggle_cell_info(self):
        self._show_id_checkbox.selected = not self._show_id_checkbox.selected
        if not self._mask_button.disabled and self._show_id_checkbox.selected:
            self.drawing_tool.show_cell_info()
            self._show_id_checkbox.icon_color = ft.Colors.WHITE
        else:
            self.drawing_tool.deactivate_cell_info()
            self._show_id_checkbox.icon_color = ft.Colors.WHITE_60
        self._show_id_checkbox.update()

    def _cell_drawn(self, lines_data: list[tuple[tuple[float, float], tuple[float, float]]]):
        if not lines_data or len(lines_data) == 0:
            return

        data_to_pass = lines_data.copy()


        self.page.run_task(self._async_draw_cell_3D, data_to_pass)

        #self.page.run_task(self._async_cell_drawn, data_to_pass)

    async def _async_cell_drawn(self, lines_data: list | np.ndarray):
        # update the mask data
        # gets the pixels that build the lines of the drawn cell

        is_new_mask = False
        if self._mask_path is None:  # currently no mask is given
            if self._image_id is None or self._seg_channel_id is None or not self._image_id in self._main_paths or not self._seg_channel_id in \
                                                                                                                       self._main_paths[
                                                                                                                           self._image_id]:
                return
            is_new_mask = True
            image_path = self._main_paths[self._image_id][self._seg_channel_id]
            directory, filename = os.path.split(image_path)
            name, _ = os.path.splitext(filename)
            mask_file_name = f"{name}{self.mask_suffix}.npy"
            self._mask_path = os.path.join(directory, mask_file_name)
            if self._image_id not in self._mask_paths:
                self._mask_paths[self._image_id] = {}
            self._mask_paths[self._image_id][self._seg_channel_id] = self._mask_path
            image_width, image_height = self.drawing_tool.get_bounds()
            if not self._image_3d:
                # 2D Case
                self._mask_data = {
                    "masks": np.zeros((image_height, image_width), dtype=np.uint16),
                    "outlines": np.zeros((image_height, image_width), dtype=np.uint16)
                }
            else:
                # 3D-Image Case (with Z-Slices)
                self._mask_data = {
                    "masks": np.zeros((self._slider_2_5d.max + 1, image_height, image_width), dtype=np.uint16),
                    "outlines": np.zeros((self._slider_2_5d.max + 1, image_height, image_width), dtype=np.uint16)
                }

        mask = self._mask_data["masks"]
        outline = self._mask_data["outlines"]

        if mask.ndim == 3:
            if self._slice_id < 0:
                raise ValueError("slice_id should be non-negative")
            mask = np.take(mask, self._slice_id, axis=0)

        if outline.ndim == 3:
            if self._slice_id < 0:
                raise ValueError("slice_id should be non-negative")
            outline = np.take(outline, self._slice_id, axis=0)

        free_id = await asyncio.to_thread(search_free_id, mask,
                                          outline, self.slice_id)  # search for the next free id in mask and outline

        # add action to undo stack to be able to delete the cell afterward
        self._undo_stack.append(("delete_action", free_id))
        self._undo_button.icon_color = ft.Colors.WHITE_60
        self._undo_button.disabled = False
        self._undo_button.update()

        temp_mask_cell = np.zeros_like(mask, dtype=np.uint8)
        # add the outline of the new mask (only the parts which not overlap with already existing cells) to outline npy array and fill the complete outline to new_cell_outline to calculate inner pixels
        if type(lines_data) is list:
            pts = np.array([[l[0][0], l[0][1]] for l in lines_data], dtype=np.int32)
            cv2.fillPoly(temp_mask_cell, [pts], 1)

        else:
            border_mask = (lines_data > 0).astype(np.uint8)
            contours, _ = cv2.findContours(border_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(temp_mask_cell, contours, -1, 1, thickness=cv2.FILLED)

        valid_area = (temp_mask_cell == 1) & (mask == 0) & (outline == 0)
        mask[valid_area] = free_id

        current_cell_full = (mask == free_id).astype(np.uint8)

        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        inner_pixels = cv2.erode(current_cell_full, kernel)

        new_outline_mask = (current_cell_full == 1) & (inner_pixels == 0)

        mask[new_outline_mask] = 0
        outline[new_outline_mask] = free_id

        mask_3d = None
        outline_3d = None
        if self._slice_id >= 0:
            mask_3d = self._mask_data["masks"]
            outline_3d = self._mask_data["outlines"]

            if mask_3d.ndim == 3:
                mask_3d[self._slice_id, :, :] = mask

            if outline_3d.ndim == 3:
                outline_3d[self._slice_id, :, :] = outline
        self._mask_data = {"masks": mask if self._slice_id == -1 else mask_3d,
                           "outlines": outline if self._slice_id == -1 else outline_3d}
        await self.update_mask_image()
        if not self._mask_image.visible:
            self._mask_image.visible = True
            self._mask_image.update()
            self._mask_button.icon_color = ft.Colors.WHITE
            self._mask_button.disabled = False
            self._mask_button.tooltip = "Hide mask"
            self._mask_button.update()
            self._show_id_checkbox.disabled = False
            if self._show_id_checkbox.selected:
                self._show_id_checkbox.icon_color = ft.Colors.WHITE
            else:
                self._show_id_checkbox.icon_color = ft.Colors.WHITE_60
            self._show_id_checkbox.update()

        self._trigger_background_save()
        self.on_mask_change(self._image_id, is_new_mask)

    async def _async_draw_cell_3D(self,lines_data: list | np.ndarray):
        # update the mask data
        # gets the pixels that build the lines of the drawn cell

        is_new_mask = False
        if self._mask_path is None:  # currently no mask is given
            if self._image_id is None or self._seg_channel_id is None or not self._image_id in self._main_paths or not self._seg_channel_id in \
                                                                                                                       self._main_paths[
                                                                                                                           self._image_id]:
                return
            is_new_mask = True
            image_path = self._main_paths[self._image_id][self._seg_channel_id]
            directory, filename = os.path.split(image_path)
            name, _ = os.path.splitext(filename)
            mask_file_name = f"{name}{self.mask_suffix}.npy"
            self._mask_path = os.path.join(directory, mask_file_name)
            if self._image_id not in self._mask_paths:
                self._mask_paths[self._image_id] = {}
            self._mask_paths[self._image_id][self._seg_channel_id] = self._mask_path
            image_width, image_height = self.drawing_tool.get_bounds()
            if not self._image_3d:
                # 2D Case
                self._mask_data = {
                    "masks": np.zeros((image_height, image_width), dtype=np.uint16),
                    "outlines": np.zeros((image_height, image_width), dtype=np.uint16)
                }
            else:
                # 3D-Image Case (with Z-Slices)
                self._mask_data = {
                    "masks": np.zeros((self._slider_2_5d.max + 1, image_height, image_width), dtype=np.uint16),
                    "outlines": np.zeros((self._slider_2_5d.max + 1, image_height, image_width), dtype=np.uint16)
                }

        mask = self._mask_data["masks"]
        outline = self._mask_data["outlines"]

        mask_3d = mask if mask.ndim == 3 else None
        outline_3d = outline if outline.ndim == 3 else None

        draw_on_all_slices = (
                self._image_3d and
                self._slice_id == -1
        )


        if mask.ndim == 3 and not draw_on_all_slices:
            if self._slice_id < 0:
                raise ValueError("slice_id should be non-negative")
            mask = np.take(mask, self._slice_id, axis=0)

        if outline.ndim == 3 and not draw_on_all_slices:
            if self._slice_id < 0:
                raise ValueError("slice_id should be non-negative")
            outline = np.take(outline, self._slice_id, axis=0)

        if draw_on_all_slices:
            mask = mask_3d[0]
            outline = outline_3d[0]

        free_id = await asyncio.to_thread(search_free_id, mask,
                                          outline,self._slice_id)  # search for the next free id in mask and outline

        # add action to undo stack to be able to delete the cell afterward
        self._undo_stack.append(("delete_action", free_id))
        self._undo_button.icon_color = ft.Colors.WHITE_60
        self._undo_button.disabled = False
        self._undo_button.update()

        temp_mask_cell = np.zeros_like(mask, dtype=np.uint8)
        # add the outline of the new mask (only the parts which not overlap with already existing cells) to outline npy array and fill the complete outline to new_cell_outline to calculate inner pixels
        if type(lines_data) is list:
            pts = np.array([[l[0][0], l[0][1]] for l in lines_data], dtype=np.int32)
            cv2.fillPoly(temp_mask_cell, [pts], 1)

        else:
            border_mask = (lines_data > 0).astype(np.uint8)
            contours, _ = cv2.findContours(border_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(temp_mask_cell, contours, -1, 1, thickness=cv2.FILLED)

        kernel = np.ones((3,3), dtype=np.uint8)

        inner_pixels = cv2.erode(temp_mask_cell, kernel)

        outline_mask = (
                (temp_mask_cell == 1) &
                (inner_pixels == 0)
        )

        fill_mask = (
                (temp_mask_cell == 1) &
                (~outline_mask)
        )

        if draw_on_all_slices:

            for z in range(mask_3d.shape[0]):
                current_mask = mask_3d[z]
                current_outline = outline_3d[z]

                affected_ids = np.unique(
                    current_mask[temp_mask_cell == 1]
                )

                affected_outline_ids = np.unique(
                    current_outline[temp_mask_cell == 1]
                )

                affected_ids = np.unique(
                    np.concatenate([affected_ids, affected_outline_ids])
                )

                affected_ids = affected_ids[affected_ids != 0]
                affected_ids = affected_ids[affected_ids != free_id]

                # refill the border if deleted in 3D mode
                affected_cells = {}

                for cid in affected_ids:
                    affected_cells[cid] = (
                            (current_mask == cid) |
                            (current_outline == cid)
                    ).copy()

                current_mask[fill_mask] = free_id
                current_outline[outline_mask] = free_id

                for cid, cell in affected_cells.items():

                    cell = cell.astype(np.uint8)

                    # remove overlap with new cell
                    cell[temp_mask_cell == 1] = 0

                    current_mask[current_mask == cid] = 0
                    current_outline[current_outline == cid] = 0

                    if np.sum(cell) == 0:
                        continue

                    inner = cv2.erode(
                        cell,
                        np.ones((3, 3), dtype=np.uint8)
                    )

                    new_outline = (
                            (cell == 1) &
                            (inner == 0)
                    )

                    new_fill = (
                            (cell == 1) &
                            (~new_outline)
                    )

                    current_mask[new_fill] = cid
                    current_outline[new_outline] = cid

        else:

            valid_area = (
                    (temp_mask_cell == 1) &
                    (mask == 0) &
                    (outline == 0)
            )

            mask[valid_area] = free_id

            current_cell_full = (mask == free_id).astype(np.uint8)

            inner_pixels = cv2.erode(current_cell_full, kernel)

            new_outline_mask = (
                    (current_cell_full == 1) &
                    (inner_pixels == 0)
            )

            mask[new_outline_mask] = 0
            outline[new_outline_mask] = free_id

        if draw_on_all_slices:
            self._mask_data = {
                "masks": mask_3d,
                "outlines": outline_3d
            }

        elif self._image_3d:

            mask_3d[self._slice_id] = mask
            outline_3d[self._slice_id] = outline

            self._mask_data = {
                "masks": mask_3d,
                "outlines": outline_3d
            }

        else:

            self._mask_data = {
                "masks": mask,
                "outlines": outline
            }

        await self.update_mask_image()
        if not self._mask_image.visible:
            self._mask_image.visible = True
            self._mask_image.update()
            self._mask_button.icon_color = ft.Colors.WHITE
            self._mask_button.disabled = False
            self._mask_button.tooltip = "Hide mask"
            self._mask_button.update()
            self._show_id_checkbox.disabled = False
            if self._show_id_checkbox.selected:
                self._show_id_checkbox.icon_color = ft.Colors.WHITE
            else:
                self._show_id_checkbox.icon_color = ft.Colors.WHITE_60
            self._show_id_checkbox.update()

        self._trigger_background_save()
        self.on_mask_change(self._image_id, is_new_mask)

    def _delete_cell(self, pos: tuple | int):
        #self.page.run_task(self._async_delete_cell, pos)
        self.page.run_task(self._async_delete_cell_3D, pos)

    async def _async_delete_cell(self, pos: tuple | int):
        image_dim ="2D"

        # delete the cell in the mask data
        if self._mask_path is None:
            return

        mask = self._mask_data["masks"]
        outline = self._mask_data["outlines"]

        if mask.ndim == 3:
            if self._slice_id < 0:
                raise ValueError("slice_id should be non-negative")
            mask = mask[self._slice_id, :, :]
            image_dim="2.5D"

        if outline.ndim == 3:
            if self._slice_id < 0:
                raise ValueError("slice_id should be non-negative")
            outline = outline[self._slice_id, :, :]

        cell_id = pos if type(pos) != tuple else _get_cell_id_from_position(pos, mask)

        if not cell_id:
            cell_id_outline = _get_cell_id_from_position(pos, outline)
            if not cell_id_outline:
                return
            cell_id = cell_id_outline

        # delete saved fluorescence cache, if cell is deleted
        cache_2d = self._fluorescence_cache.fluorescence_cache.get(image_dim, {})
        slice_cache = cache_2d.get(self._channel_id, {})

        condition = (
                (
                        self._slice_id in slice_cache
                        and cell_id in slice_cache[self._slice_id]
                )
                or (
                        None in slice_cache
                        and cell_id in slice_cache[None]
                )
        )
        if condition:
            self._fluorescence_cache.fluorescence_cache[image_dim][self._channel_id][
                self._slice_id if self._slice_id != -1 else None].pop(cell_id)

        # Update the mask and outline (delete the cell)
        cell_mask = (mask == cell_id)
        cell_outline = (outline == cell_id)
        # add line data to the undo stack to draw the cell later out of the line
        self._undo_stack.append(("draw_action", cell_outline.copy()))
        self._undo_button.icon_color = ft.Colors.WHITE_60
        self._undo_button.disabled = False
        self._undo_button.update()
        # ------

        mask[cell_mask] = 0
        outline[cell_outline] = 0
        if self._shifting_check_box.selected:
            await asyncio.to_thread(mask_shifting, self._mask_data, cell_id, self._slice_id)
            self._fluorescence_cache.clear()

        await self.update_mask_image()
        self._trigger_background_save()
        self.on_mask_change(self._image_id, False)

    async def _async_delete_cell_3D(self, pos: tuple | int):
        # delete the cell in the mask data
        if self._mask_path is None:
            return

        mask = self._mask_data["masks"]
        outline = self._mask_data["outlines"]

        delete_cell_on_all_slices = (
                self._image_3d and
                self._slice_id == -1
        )

        image_dim = "3D" if delete_cell_on_all_slices else ("2D" if not self._image_3d else "2.5D")

        if mask.ndim == 3 and not delete_cell_on_all_slices:
            if self._slice_id < 0:
                raise ValueError("slice_id should be non-negative")
            mask = mask[self._slice_id, :, :]

        if outline.ndim == 3 and not delete_cell_on_all_slices:
            if self._slice_id < 0:
                raise ValueError("slice_id should be non-negative")
            outline = outline[self._slice_id, :, :]

        cell_id = pos if type(pos) != tuple else _get_cell_id_from_position(pos, mask)

        if cell_id is None:
            cell_id_outline = _get_cell_id_from_position(pos, outline)
            if cell_id_outline is None:
                return
            cell_id = cell_id_outline

        # Update the mask and outline (delete the cell)
        if delete_cell_on_all_slices:
            cell_id = np.unique(cell_id)
            cell_id = next((x for x in cell_id if x != 0), None)

            for image_slice in range(mask.shape[0]):
                current_mask =mask[image_slice]
                current_outline =outline[image_slice]

                cell_mask = ( current_mask == cell_id )
                cell_outline = (current_outline == cell_id)

                current_mask[cell_mask] = 0
                current_outline[cell_outline] = 0
        else:
            cell_mask = (mask == cell_id)
            cell_outline = (outline == cell_id)

            mask[cell_mask] = 0
            outline[cell_outline] = 0

        # delete saved fluorescence cache, if cell is deleted
        cache_2d = self._fluorescence_cache.fluorescence_cache.get(image_dim, {})
        slice_cache = cache_2d.get(self._channel_id, {})

        print("cellid:",cell_id)
        condition = (
                (
                        self._slice_id in slice_cache
                        and cell_id in slice_cache[self._slice_id]
                )
                or (
                        None in slice_cache
                        and cell_id in slice_cache[None]
                )
        )
        if condition:
            self._fluorescence_cache.fluorescence_cache[image_dim][self._channel_id][
                self._slice_id if self._slice_id != -1 else None].pop(cell_id)

        # add line data to the undo stack to draw the cell later out of the line
        self._undo_stack.append(("draw_action", cell_outline.copy()))
        self._undo_button.icon_color = ft.Colors.WHITE_60
        self._undo_button.disabled = False
        self._undo_button.update()
        # ------

        if self._shifting_check_box.selected:
            await asyncio.to_thread(mask_shifting, self._mask_data, cell_id, self._slice_id)
            self._fluorescence_cache.clear()

        await self.update_mask_image()
        self._trigger_background_save()
        self.on_mask_change(self._image_id, False)

    def _trigger_background_save(self):
        current_path = self._mask_path
        current_data = self._mask_data
        self.page.run_task(self._save_async,current_path,current_data)

    def _save_async(self,current_path,current_data):
        if current_path is None or current_data is None:
            return

        if self._save_lock is None:
            self._save_lock = asyncio.Lock()

        data_copy = copy.deepcopy(current_data)
        async def save():
            async with self._save_lock:
                await asyncio.to_thread(np.save, current_path, data_copy, allow_pickle=True)

        asyncio.run(save())

    def delete_mask(self):
        def cancel_dialog(a):
            cupertino_alert_dialog.open = False
            a.control.page.update()

        def ok_dialog(a):
            cupertino_alert_dialog.open = False
            a.control.page.update()
            self.reset_mask()

        cupertino_alert_dialog = ft.CupertinoAlertDialog(
            title=ft.Text("Delete Entire Mask"),
            content=ft.Text("Are you sure you want to delete all drawn cells on this image?\n\n"
                            "The underlying mask file will be deleted. "
                            "You can always start over by drawing new cells, but the current state cannot be recovered with undo operations."),
            actions=[
                ft.CupertinoDialogAction(
                    "Cancel", default=True, on_click=cancel_dialog
                ),
                ft.CupertinoDialogAction("Ok", destructive=True, on_click=lambda a: ok_dialog(a)),
            ],
        )
        self.page.overlay.append(cupertino_alert_dialog)
        cupertino_alert_dialog.open = True
        self.page.update()

    async def redo_stack(self, e):
        if len(self._redo_stack) == 0:
            return
        self._undo_button.icon_color = ft.Colors.WHITE_60
        self._undo_button.disabled = False
        self._undo_button.update()
        first_list_item = self._redo_stack.pop()

        if first_list_item[0] == "delete_action":
            await self._async_delete_cell_3D(first_list_item[1])
        elif first_list_item[0] == "draw_action":
            await self._async_draw_cell_3D(first_list_item[1])
        else:
            raise KeyError("no valid action for redo button")

        if len(self._redo_stack) == 0:
            self._redo_button.icon_color = ft.Colors.BLACK_12
            self._redo_button.disabled = True
            self._redo_button.update()
        if len(self._undo_stack) == 0:
            self._undo_button.icon_color = ft.Colors.BLACK_12
            self._undo_button.disabled = True
            self._undo_button.update()

    async def undo_stack(self, e):
        if len(self._undo_stack) == 0:
            return

        self._redo_button.icon_color = ft.Colors.WHITE_60
        self._redo_button.disabled = False
        self._redo_button.update()
        first_list_item = self._undo_stack.pop()
        if first_list_item[0] == "delete_action":
            await self._async_delete_cell_3D(first_list_item[1])
        elif first_list_item[0] == "draw_action":
            await self._async_draw_cell_3D(first_list_item[1])
        else:
            raise KeyError("no valid action for undo button")

        self._redo_stack.append(self._undo_stack.pop())
        if len(self._redo_stack) == 0:
            self._redo_button.icon_color = ft.Colors.BLACK_12
            self._redo_button.disabled = True
            self._redo_button.update()
        if len(self._undo_stack) == 0:
            self._undo_button.icon_color = ft.Colors.BLACK_12
            self._undo_button.disabled = True
            self._undo_button.update()

    def show_ids_and_value(self, pos: tuple):
        if self._mask_path is None or self._mask_button.icon_color == ft.Colors.WHITE_60 or self._mask_button.icon_color == ft.Colors.BLACK_12:
            return

        mask = self._mask_data["masks"]

        # if hovered over cell, get cell id
        cell_id = _get_cell_id_from_position(pos, mask)

        if cell_id is None or cell_id == 0:
            self._id_info.visible = False
            self._id_info.update()
            return

        # load fluorescence value from cache

        cell_value = self._fluorescence_cache.get_fluorescence_value(cell_id, mask, np.array(
            self._image_cache.get_image(self._main_paths[self._image_id][self._channel_id])), "2D", self._channel_id,
                                                                     self._slice_id)

        # show id and value in canvas
        if self._show_id_checkbox.selected:
            self._id_info.content.rows = [
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(f"{cell_id}", color=ft.Colors.WHITE)),
                        ft.DataCell(ft.Text(f"{cell_value:.2f}", color=ft.Colors.WHITE)),
                    ]
                ),
            ]
            self._id_info.visible = True
            self._id_info.update()

    def show_ids_and_value_3d(self, pos: tuple):
        if self._mask_path is None or self._mask_button.icon_color == ft.Colors.WHITE_60 or self._mask_button.icon_color == ft.Colors.BLACK_12:
            return

        mask = self._mask_data["masks"]

        if mask.ndim == 3 and self._user_2_5d:
            if self._slice_id < 0:
                raise ValueError("slice_id should be non-negative")
            slice_mask = mask[self._slice_id, :, :]
            cell_id = _get_cell_id_from_position(pos, slice_mask)
            if cell_id is None or cell_id == 0:
                self._id_info.visible = False
                self._id_info.update()
                return

            cell_value = self._fluorescence_cache.get_fluorescence_value(cell_id, mask, np.array(
                self._image_cache.get_image(self._main_paths[self._image_id][self._channel_id])), "2.5D",
                                                                         self._channel_id, self._slice_id)
            values = [ft.DataRow(
                cells=[
                    ft.DataCell(ft.Text(f"{cell_id}", color=ft.Colors.WHITE)),
                    ft.DataCell(ft.Text(f"{cell_value:.2f}", color=ft.Colors.WHITE)),
                ]
            )]
        else:
            cell_id = _get_cell_id_from_position(pos, mask)

            cell_id = np.unique(cell_id)
            values = []
            for cellid in cell_id:
                if cellid != 0:
                    cell_value = self._fluorescence_cache.get_fluorescence_value(cellid, mask, np.array(
                        self._image_cache.get_image(self._main_paths[self._image_id][self._channel_id])), "3D",
                                                                                 self._channel_id, self._slice_id)

                    values.append(
                        ft.DataRow(
                            cells=[
                                ft.DataCell(ft.Text(f"{cellid}", color=ft.Colors.WHITE)),
                                ft.DataCell(ft.Text(f"{cell_value:.2f}", color=ft.Colors.WHITE)),
                            ]
                        )
                    )
            if len(values) == 0:
                self._id_info.visible = False
                self._id_info.update()
                return

        if self._show_id_checkbox.selected:
            self._id_info.content.rows = values

            self._id_info.visible = True
            self._id_info.update()

    def _handle_show_ids(self, pos: tuple):
        if self._image_3d:
            self.show_ids_and_value_3d(pos)
        else:
            self.show_ids_and_value(pos)

    def disable_editing_without_update(self):
        self._edit_allowed = False
        self._edit_button.icon_color = ft.Colors.BLACK12
        self._edit_button.disabled = True
        self.drawing_tool.deactivate_drawing()
        self._delete_button.icon_color = ft.Colors.BLACK_12
        self._delete_button.disabled = True
        self.drawing_tool.deactivate_delete()
        self._redo_button.disabled = True
        self._undo_button.disabled = True
        self._redo_button.icon_color = ft.Colors.BLACK_12
        self._undo_button.icon_color = ft.Colors.BLACK_12
        self._delete_mask_button.icon_color = ft.Colors.BLACK_12
        self._delete_mask_button.disabled = True

    def reset_mask(self):
        if self._mask_path is not None:
            if os.path.exists(self._mask_path):
                os.remove(self._mask_path)
            if self._mask_paths and self._image_id in self._mask_paths:
                self._mask_paths[self._image_id].pop(self._seg_channel_id, None)
            self._mask_path = None
            self._mask_data = None
            self._redo_stack.clear()
            self._undo_stack.clear()
            self._redo_button.disabled = True
            self._undo_button.disabled = True
            self._redo_button.icon_color = ft.Colors.BLACK_12
            self._undo_button.icon_color = ft.Colors.BLACK_12
            self._redo_button.update()
            self._undo_button.update()
            self._show_id_checkbox.disabled = True
            self._show_id_checkbox.icon_color = ft.Colors.BLACK_12
            self._show_id_checkbox.selected = False
            self.drawing_tool.deactivate_cell_info()
            self._id_info.visible = False
            self._id_info.update()
            self._show_id_checkbox.update()
            self.page.run_task(self.update_mask_image)
            self.on_mask_change(self._image_id, True)

    def check_edit_allowed(self):
        if self._edit_allowed and not(self._image_id is None or self._seg_channel_id is None or not self._image_id in self._main_paths or not self._seg_channel_id in self._main_paths[self._image_id]):
            return True
        else:
            return False
