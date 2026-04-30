import asyncio
import base64
import math
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
from drawing_util import bresenham_line, search_free_id, trace_contour, fill_polygon_from_outline, find_border_pixels, \
    mask_shifting, rgb_to_hex


def load_image(image,auto_adjust=False,get_slice=-1,brightness=1.0, contrast=1.0):
    shape = list(image.shape)
    check = image.ndim == 3
    if check:
        if not get_slice == -1:
            image = image[:, :, get_slice]
        else:
            image = np.max(image, axis=2)

    if auto_adjust:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    elif brightness != 1.0 or contrast != 1.0:
        img = Image.fromarray(image)
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(brightness)
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(contrast)

        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8'),shape,check

    _, buffer = cv2.imencode('.png', image)

    return base64.b64encode(buffer).decode('utf-8'), shape, check


def convert_npy_to_canvas(mask, outline, mask_color, outline_color, opacity, slice_id=-1):
    """
    handles the conversion of the given file data

    Args:
        mask= the mask data stored in the numpy directory
        outline= the outline data stored in the numpy directory
    """
    if mask.ndim == 3:
        if slice_id >= 0:
            mask = mask[slice_id, :, :]
        else:
            mask = mask.any(axis=0)

    mask = mask != 0

    if outline.ndim == 3:
        if slice_id >= 0:
            outline = outline[slice_id, :, :]
        else:
            outline = outline.any(axis=0)

    outline = outline != 0

    image_mask = np.zeros(shape=(mask.shape[0], mask.shape[1], 4), dtype=np.uint8)

    if mask.any():
        r,g,b = mask_color
        image_mask[mask] = (r, g, b, opacity)

    if outline.any():
        r, g, b = outline_color
        image_mask[outline] = (r, g, b, 255)

    im= Image.fromarray(image_mask, mode="RGBA")

    #saves the image as a image(base64)
    buffer= BytesIO()
    im.save(buffer, format="PNG", compress_level=1)

    buffer.seek(0)
    image_base_64= base64.b64encode(buffer.getvalue()).decode('utf-8')

    #saves the created output image.
    return image_base_64


def _get_cell_id_from_position(position, mask):
    """
    Get the cell ID from the clicked position.
    """
    x, y = int(position[0]), int(position[1])
    if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
        return mask[y, x]
    return None

class FluorescenceCache:
    def __init__(self, max_values=30):
        self.fluorescence_cache = OrderedDict()
        self._max_values =max_values

    def clear(self):
        self.fluorescence_cache.clear()

    def get_fluorescence_value(self, cell_id, mask, np_image, channel, zslice=None):
        if zslice == -1 :
            zslice =None

        if channel not in self.fluorescence_cache:
            self.fluorescence_cache[channel] = OrderedDict()
        if zslice not in self.fluorescence_cache[channel]:
            self.fluorescence_cache[channel][zslice] =OrderedDict()

        if cell_id in self.fluorescence_cache[channel][zslice]:
            self.fluorescence_cache[channel][zslice].move_to_end(cell_id)
            self.fluorescence_cache[channel].move_to_end(zslice)
            self.fluorescence_cache.move_to_end(channel)
            return self.fluorescence_cache[channel][zslice][cell_id]

        if len(self.fluorescence_cache[channel][zslice]) > self._max_values:
            self.fluorescence_cache[channel][zslice].popitem(last=False)

        cell_mask = mask == cell_id
        val = float(np.mean(np_image[cell_mask]))

        self.fluorescence_cache[channel][zslice][cell_id] = val
        self.fluorescence_cache[channel][zslice].move_to_end(cell_id)

        return val


class ImageCache:
    def __init__(self, max_images=10):
        self.cache = OrderedDict()
        self._max_images = max_images

    def get_image(self, path):
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        else:
            image = tifffile.imread(path)
            self.add_image(path,image)
            return image

    def add_image(self, path, data):
        self.cache[path] = data
        self.cache.move_to_end(path)
        if len(self.cache) > self._max_images:
            self.cache.popitem(last=False)

    def clear(self):
        self.cache.clear()

class ImageEditingView(ft.Card):
    def __init__(self,on_mask_change: typing.Callable[[str,bool], None] = None):
        super().__init__()
        self._mask_paths = None
        self._main_paths = None
        self._mask_path =None#Could set a mask_path for TESTING
        self._mask_data = None#np.load(Path(self._mask_path), allow_pickle=True).item()
        self._slice_id = -1
        self._image_3d = False
        self._image_id = None
        self._channel_id = None
        self._seg_channel_id = None
        self._save_task = None
        self._image_cache = ImageCache()
        self._fluorescence_cache = FluorescenceCache()
        self._running_tasks = set()
        self.brightness = 1.0
        self.contrast = 1.0
        self._on_click = False
        self.auto_adjust = False
        self.mask_color = (255, 0, 0)
        self.outline_color = (0, 255, 0)
        self.mask_opacity = 128
        self._user_2_5d = False
        self.on_mask_change = on_mask_change or (lambda x: None)
        self.mask_suffix = "_seg"
        self.expand=True
        self._redo_stack = []
        self._undo_stack = []
        self._edit_allowed = True
        self._mask_image = ft.Image(src="Placeholder", fit=ft.BoxFit.CONTAIN, visible=False,gapless_playback=True,expand=True)
        self._main_image = ft.Image(src="Placeholder", fit=ft.BoxFit.CONTAIN,visible=False,gapless_playback=True,expand=True)
        self.drawing_tool = DrawingTool(on_cell_drawn=self._cell_drawn, on_cell_deleted=self._delete_cell,on_show_ids=self.show_ids_and_value)

        self._mask_button = ft.IconButton(icon=ft.Icons.REMOVE_RED_EYE, icon_color=ft.Colors.BLACK12,
                                          style=ft.ButtonStyle(
                                              shape=ft.RoundedRectangleBorder(radius=12),), on_click=lambda e: self._show_mask(),
                                          tooltip="Show Mask", hover_color=ft.Colors.WHITE12, disabled=True)
        self._edit_button = ft.IconButton(icon=ft.Icons.BRUSH, icon_color=ft.Colors.BLACK_12,
                                          style=ft.ButtonStyle(
                                              shape=ft.RoundedRectangleBorder(radius=12), ),disabled=True,
                                          tooltip="Draw mode", hover_color=ft.Colors.WHITE_12,on_click=lambda e:self._toggle_draw())
        self._delete_button = ft.IconButton(icon=ft.Icons.CLEAR, icon_color=ft.Colors.BLACK_12,
                                          style=ft.ButtonStyle(
                                              shape=ft.RoundedRectangleBorder(radius=12), ),disabled=True,
                                          tooltip="Delete mode", hover_color=ft.Colors.WHITE12,on_click=lambda e: self._toggle_delete())
        self._delete_mask_button = ft.IconButton(icon=ft.Icons.DELETE_FOREVER, icon_color=ft.Colors.WHITE_60,
                                            style=ft.ButtonStyle(
                                                shape=ft.RoundedRectangleBorder(radius=12), ),
                                            tooltip="Delete the complete mask", hover_color=ft.Colors.WHITE12,
                                            on_click=lambda e: self.delete_mask())
        self._redo_button = ft.IconButton(icon=ft.Icons.REDO_SHARP, icon_color=ft.Colors.BLACK_12,
                                            style=ft.ButtonStyle(
                                                shape=ft.RoundedRectangleBorder(radius=12), ),
                                            tooltip="Redo action", hover_color=ft.Colors.WHITE_12,
                                            on_click=lambda e: self.redo_stack(e),disabled=True)

        self._undo_button = ft.IconButton(icon=ft.Icons.UNDO_SHARP, icon_color=ft.Colors.BLACK_12,
                                            style=ft.ButtonStyle(
                                                shape=ft.RoundedRectangleBorder(radius=12), ),
                                            tooltip="Undo action", hover_color=ft.Colors.WHITE12,
                                            on_click=lambda e: self.undo_stack(e),disabled=True)

        #controls for visible cell id and value, when hovered over the cell mask
        self._show_id_checkbox = ft.IconButton(
            icon=ft.CupertinoIcons.NUMBER_CIRCLE_FILL,
            icon_color=ft.Colors.BLACK_12,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=12), ),
            hover_color=ft.Colors.WHITE12,
            selected=False,
            disabled=True,
            on_click=lambda e: self._toggle_cell_info(),
        )
        self._id_info = ft.Container(
            content=ft.Text(
                "",
                color=ft.Colors.WHITE,
                size=14,
                weight=ft.FontWeight.BOLD
            ),
            bgcolor=ft.Colors.BLACK54,
            padding=8,
            border_radius=10,
            visible=False,
        )

        self._slider_2_5d = ft.Slider(
            min=0, max=100, divisions=None, label="Slice: {value}",value=0,
            opacity=1.0 if self._user_2_5d else 0.0, height=20,width=170,
            active_color=ft.Colors.WHITE60, thumb_color=ft.Colors.WHITE, disabled=True,
            animate_opacity=ft.Animation(duration=600, curve=ft.AnimationCurve.LINEAR_TO_EASE_OUT),
            on_change=lambda e: self._slider2_5d_change()
        )
        self._slider_2d = ft.CupertinoSlidingSegmentedButton(
            selected_index=0 if not self._user_2_5d else 1,
            thumb_color=ft.Colors.WHITE,
            bgcolor=ft.Colors.WHITE60,
            padding=ft.padding.symmetric(0, 0),
            controls=[
                ft.Text("2D", color=ft.Colors.BLACK,weight=ft.FontWeight.BOLD),
                ft.Text("2.5D", color=ft.Colors.BLACK,weight=ft.FontWeight.BOLD)
            ],
            on_change=lambda e: self._slider2d_update(e)
        )
        self._shifting_check_box = ft.IconButton(
            icon=ft.Icon(ft.Icons.FORMAT_LIST_NUMBERED,color=ft.Colors.WHITE60),
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=12),),
            hover_color=ft.Colors.WHITE12,
            selected_icon=ft.Icon(ft.Icons.FORMAT_LIST_NUMBERED,color=ft.Colors.WHITE),
            selected=False,
            on_click=lambda e: self._toggle_shifting(e),
        )
        self.control_tools = ft.Container(ft.Container(ft.Row(
                [   self._undo_button,
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
                                value_indicator_text_style=ft.TextStyle(color=ft.Colors.BLACK, size=15,weight=ft.FontWeight.BOLD),
                            )
                        ),
                        dark_theme=ft.Theme(
                            slider_theme=ft.SliderTheme(
                                value_indicator_text_style=ft.TextStyle(color=ft.Colors.BLACK, size=15,weight=ft.FontWeight.BOLD),
                            )
                        ),
                    ),
                    self._delete_mask_button,
                    self._show_id_checkbox,
                ], spacing=2,alignment=ft.MainAxisAlignment.CENTER,height=38,
            ), bgcolor=ft.Colors.BLUE_ACCENT, expand=True, border_radius=ft.border_radius.vertical(top=0, bottom=12),

                ))
        self.image_stack = ft.InteractiveViewer(content=ft.Stack([self._main_image,
                                                                  self._mask_image,
                                                                  self.drawing_tool,
                                                                  ], expand=True), expand=True)

        self.content = ft.Stack([ft.Column(controls=[ft.Container(self.image_stack,alignment=ft.Alignment.CENTER,expand=True),self.control_tools],spacing=0),
                                                                    ft.Container(
                                                                      content=self._id_info,
                                                                      right=15,
                                                                      top=15,
                                                                  )])

    def set_mask_paths(self, mask_paths: list):
        self._mask_paths = mask_paths

    def set_main_paths(self, main_paths: list):
        self._main_paths = main_paths

    def set_colors(self, mask_color, outline_color, opacity):
        self.drawing_tool.draw_color= rgb_to_hex(outline_color)
        self.mask_color = mask_color
        self.outline_color = outline_color
        self.mask_opacity = opacity
        self.update_mask_image()

    def reset_image(self):
        self._main_image.src = "Placeholder"
        self._main_image.visible = False
        self._main_image.update()
        self._seg_channel_id = None
        self._image_id = None
        self._mask_path = None
        self._mask_data = None
        self._mask_image.src = "Placeholder"
        self._mask_image.visible = False
        self._mask_image.update()
        self._mask_button.tooltip = "Show mask"
        self._mask_button.icon_color = ft.Colors.BLACK12
        self._mask_button.disabled = True
        self._mask_button.update()
        self._edit_button.icon_color = ft.Colors.BLACK12
        self._edit_button.disabled = True
        self._edit_button.update()
        self._delete_button.icon_color = ft.Colors.BLACK_12
        self._delete_button.disabled = True
        self._delete_button.update()
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
        self._redo_button.update()
        self._undo_button.update()
        self._image_cache.clear()
        self._show_id_checkbox.disabled = True
        self._show_id_checkbox.icon_color = ft.Colors.BLACK_12
        self._show_id_checkbox.selected = False
        self._show_id_checkbox.update()
        self.drawing_tool.deactivate_cell_info()
        self._id_info.visible = False
        self._id_info.update()
        self._fluorescence_cache.clear()
        self.cancel_all_tasks()

    def select_image(self, img_id, channel_id,seg_channel_id, on_click=False):
        if self._seg_channel_id != seg_channel_id or self._image_id != img_id:
            self._load_mask_image(img_id, seg_channel_id)
        self._image_id = img_id
        self._on_click = on_click
        self._channel_id = channel_id
        self._seg_channel_id = seg_channel_id
        self._load_main_image(img_id, channel_id)
        #reset undo/redo when a new image is selected
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

    def _slider2d_update(self,e):
        if int(e.data) == 1:
            self._slider_2_5d.opacity = 1.0
            self.user_2_5d = True
        else:
            self._slider_2_5d.opacity = 0.0
            self.user_2_5d = False

        self._slider2_5d_change()
        self._slider_2_5d.update()

    def _slider2_5d_change(self):
        if self.user_2_5d:
            self._slice_id = int(self._slider_2_5d.value)
        else:
            self._slice_id = -1

        if self._main_image.src != "Placeholder":
            self._load_main_image(self._image_id,self._channel_id)
            self.update_mask_image()
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

    async def _adjust_image_async(self, path, brightness,contrast):
        return await asyncio.to_thread(load_image,self._image_cache.get_image(path),False, self._slice_id,brightness,contrast)

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


    async def update_main_image_with_brightness_contrast(self,path):
        task = asyncio.create_task(self._update_main_image(path))
        self._running_tasks.add(task)
        try:
            await task
        except asyncio.CancelledError:
            return
        finally:
            self._running_tasks.discard(task)

    def _load_main_image_with_path(self,path):
        self.cancel_all_tasks()
        src, shape, img_3d = load_image(self._image_cache.get_image(path), auto_adjust=self.auto_adjust, get_slice=self._slice_id,brightness=self.brightness,contrast=self.contrast)
        self._main_image.src = src
        self._main_image.visible = True
        self.drawing_tool.set_bounds(shape[1],shape[0])
        self._main_image.update()
        if img_3d:
            self._image_3d = True
            if self._slider_2_5d.opacity == 1.0 and self._edit_allowed:
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
                self._edit_button.icon_color = ft.Colors.BLACK12
                self._edit_button.disabled = True
                self._edit_button.update()
                self.drawing_tool.deactivate_drawing()
                self._delete_button.icon_color = ft.Colors.BLACK_12
                self._delete_button.disabled = True
                self._delete_button.update()
                self.drawing_tool.deactivate_delete()
                self._show_id_checkbox.disabled = True
                self._show_id_checkbox.icon_color = ft.Colors.BLACK_12
                self._show_id_checkbox.selected = False
                self.drawing_tool.deactivate_cell_info()
                self._id_info.visible = False
                self._id_info.update()
                self._show_id_checkbox.update()
                self.drawing_tool.deactivate_cell_info()
            self._slider_2_5d.value = 0 if shape[-2] - 1 < self._slider_2_5d.value else self._slider_2_5d.value
            self._slider_2_5d.max = shape[2] - 1
            self._slider_2_5d.divisions = shape[2] - 2
            self._slider_2_5d.disabled = False
            self._slider_2_5d.update()
        else:
            self._image_3d = False
            if self._edit_allowed:
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
            self._slider_2_5d.value = 0
            self._slice_id = 0
            self._slider_2_5d.max = 1
            self._slider_2_5d.divisions = None
            self._slider_2_5d.disabled = True
            self._slider_2_5d.update()
        return

    def _load_mask_image(self, img_id, seg_channel_id):
        if self._mask_paths is not None:
            if img_id in self._mask_paths:
                if seg_channel_id in self._mask_paths[img_id]:
                    new_path = self._mask_paths[img_id][seg_channel_id]
                    if new_path != self._mask_path:
                        self._mask_data = np.load(
                            Path(self._mask_paths[img_id][seg_channel_id]),allow_pickle=True).item()
                        self._mask_path = new_path
                        self._mask_data["masks"] = self._mask_data["masks"].astype(np.uint16)
                        self._mask_data["outlines"] = self._mask_data["outlines"].astype(np.uint16)

                    self._mask_image.src = convert_npy_to_canvas(self._mask_data["masks"], self._mask_data["outlines"], self.mask_color, self.outline_color, self.mask_opacity, slice_id=self._slice_id)
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

    def update_mask_image(self):
        if self._mask_path is not None:
            self._update_mask_image()
        elif self._mask_paths is not None and self._image_id in self._mask_paths and self._seg_channel_id in self._mask_paths[self._image_id] and self._mask_paths[self._image_id][self._seg_channel_id] is not None:
            self._mask_path = self._mask_paths[self._image_id][self._seg_channel_id]
            self._mask_data = np.load(Path(self._mask_path), allow_pickle=True).item()
            self._mask_data["masks"] = self._mask_data["masks"].astype(np.uint16)
            self._mask_data["outlines"] = self._mask_data["outlines"].astype(np.uint16)
            self._update_mask_image()
        else:
            self._mask_image.src = "Placeholder"
            self._mask_image.visible = False
            self._mask_image.update()
            self._mask_button.tooltip = "Show mask"
            self._mask_button.icon_color = ft.Colors.BLACK12
            self._mask_button.disabled = True
            self._mask_button.update()
            self._show_id_checkbox.disabled = True
            self._show_id_checkbox.icon_color = ft.Colors.BLACK_12
            self._show_id_checkbox.selected = False
            self._show_id_checkbox.update()
            self.drawing_tool.deactivate_cell_info()
            self._id_info.visible = False
            self._id_info.update()

    def _update_mask_image(self):
        if self._mask_data is None:
            return
        if not self._mask_image.visible:
            self._mask_button.icon_color = ft.Colors.WHITE60
            self._mask_button.tooltip = "Show mask"
            self._mask_button.disabled = False
            self._mask_button.update()
            self._show_id_checkbox.disabled = False
            if self._show_id_checkbox.selected:
                self._show_id_checkbox.icon_color = ft.Colors.WHITE
            else:
                self._show_id_checkbox.icon_color = ft.Colors.WHITE_60
            self._show_id_checkbox.update()
        mask = self._mask_data["masks"]
        outline = self._mask_data["outlines"]
        self._mask_image.src = convert_npy_to_canvas(mask, outline, self.mask_color, self.outline_color,
                                                     self.mask_opacity, slice_id=self._slice_id)
        self._mask_image.update()

    def _show_mask(self):
        self._mask_image.visible = not self._mask_image.visible
        self._mask_image.update()
        self._mask_button.icon_color = ft.Colors.WHITE if self._mask_image.visible else ft.Colors.WHITE60
        self._mask_button.tooltip="Hide mask" if self._mask_image.visible else "Show mask"
        self._mask_button.update()

    def _toggle_draw(self):
        self._edit_button.icon_color = ft.Colors.WHITE if self._edit_button.icon_color==ft.Colors.WHITE_60 else ft.Colors.WHITE60
        self._edit_button.update()
        if self._edit_button.icon_color==ft.Colors.WHITE:
            self._delete_button.icon_color = ft.Colors.WHITE60
            self._delete_button.update()
            self.drawing_tool.draw()
        else:
            self.drawing_tool.deactivate_drawing()

    def _toggle_delete(self):
        self._delete_button.icon_color = ft.Colors.WHITE if self._delete_button.icon_color == ft.Colors.WHITE_60 else ft.Colors.WHITE60
        self._delete_button.update()
        if self._delete_button.icon_color == ft.Colors.WHITE:
            if not self._edit_button.disabled:
                self._edit_button.icon_color = ft.Colors.WHITE60
                self._edit_button.update()
                self.drawing_tool.delete()
        else:
            self.drawing_tool.deactivate_delete()

    def _toggle_shifting(self,e):
        e.control.selected = not e.control.selected
        if e.control.selected:
            e.control.tooltip = "Shifting IDs: ON \n(Shifts the IDs when a mask is deleted to restore an order without gaps.)"
        else:
            e.control.tooltip = "Shifting IDs: OFF \n(Deleted masks will leave gaps in the order of the IDs. No shifting will occur.)"

        e.control.update()

    def _toggle_cell_info(self):
        self._show_id_checkbox.selected = not self._show_id_checkbox.selected
        if not self._mask_button.disabled and self._show_id_checkbox.selected:
            self.drawing_tool.show_cell_info()
            self._show_id_checkbox.icon_color = ft.Colors.WHITE
        else:
            self.drawing_tool.deactivate_cell_info()
            self._show_id_checkbox.icon_color = ft.Colors.WHITE_60
        self._show_id_checkbox.update()

    def _cell_drawn(self, lines_data: list | np.ndarray):
        #update the mask data
        # gets the pixels that build the lines of the drawn cell
        is_new_mask = False
        if self._mask_path is None: #currently no mask is given
            if self._image_id is None or self._seg_channel_id is None or not self._image_id in self._main_paths or not self._seg_channel_id in self._main_paths[self._image_id]:
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
                #2D Case
                self._mask_data = {
                    "masks": np.zeros((image_height, image_width), dtype=np.uint16),
                    "outlines": np.zeros((image_height, image_width), dtype=np.uint16)
                }
            else:
                #3D-Image Case (with Z-Slices)
                self._mask_data = {
                    "masks": np.zeros((self._slider_2_5d.max + 1, image_height, image_width), dtype=np.uint16),
                    "outlines": np.zeros((self._slider_2_5d.max + 1, image_height, image_width), dtype=np.uint16)
                }

        line_pixels = set()
        if type(lines_data) is list:
            for line in lines_data:
                pixels = bresenham_line(line[0], line[1])  # Calculates the pixels along the line
                line_pixels.update(pixels)

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

        free_id = search_free_id(mask, outline)  # search for the next free id in mask and outline
        # add action to undo stack to be able to delete the cell afterward
        self._undo_stack.append(("delete_action", free_id))
        self._undo_button.icon_color = ft.Colors.WHITE_60
        self._undo_button.disabled = False
        self._undo_button.update()

        # add the outline of the new mask (only the parts which not overlap with already existing cells) to outline npy array and fill the complete outline to new_cell_outline to calculate inner pixels
        new_cell_outline = np.zeros_like(outline, dtype=np.uint16)
        if type(lines_data) is list:
            for x, y in line_pixels:
                if 0 <= x < outline.shape[1] and 0 <= y < outline.shape[0]:
                    new_cell_outline[y, x] = 1
                    if outline[y, x] == 0 and mask[y, x] == 0:
                        outline[y, x] = free_id
        else:
            new_cell_outline = lines_data

        # Traces the outline of the new cell and fills the mask based on the outline
        contour = trace_contour(new_cell_outline)
        new_mask = fill_polygon_from_outline(contour, mask.shape)  # gets the inner pixels of the new cell
        mask[(new_mask == 1) & (mask == 0) & (
                    outline == 0)] = free_id  # adds them to the npy if they not overlap with the already existing cells

        # search if inline pixels (mask) have no outline, if the pixel have no outline neighbor make them to outline and delete them from mask
        new_border_pixels = find_border_pixels(mask, outline, free_id)
        for y, x in new_border_pixels:
            if 0 <= x < outline.shape[1] and 0 <= y < outline.shape[0]:
                mask[y, x] = 0
                outline[y, x] = free_id

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

        self.update_mask_image()
        if not self._mask_image.visible:
            self._mask_image.visible = True
            self._mask_image.update()
            self._mask_button.icon_color = ft.Colors.WHITE
            self._mask_button.tooltip = "Hide mask"
            self._mask_button.update()

        self._trigger_background_save()
        self.on_mask_change(self._image_id,is_new_mask)

    def _delete_cell(self, pos: tuple | int):

        #delete the cell in the mask data
        if self._mask_path is None:
            return

        mask = self._mask_data["masks"]
        outline = self._mask_data["outlines"]

        if mask.ndim == 3:
            if self._slice_id < 0:
                raise ValueError("slice_id should be non-negative")
            mask = mask[self._slice_id, :, :]

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

        #delete saved fluorescence cache, if cell is deleted
        self._fluorescence_cache.fluorescence_cache[self._channel_id][self._slice_id if self._slice_id != -1 else None].pop(cell_id)

        # Update the mask and outline (delete the cell)
        cell_mask = (mask == cell_id)
        cell_outline = (outline == cell_id)
        # add line data to the undo stack to draw the cell later out of the line
        self._undo_stack.append(("draw_action", cell_outline.copy()))
        self._undo_button.icon_color = ft.Colors.WHITE_60
        self._undo_button.disabled = False
        self._undo_button.update()
        #------

        mask[cell_mask] = 0
        outline[cell_outline] = 0
        if self._shifting_check_box.selected:
            mask_shifting(self._mask_data, cell_id, self._slice_id)
            self._fluorescence_cache.clear()

        self.update_mask_image()
        self._trigger_background_save()
        self.on_mask_change(self._image_id,False)

    def _trigger_background_save(self):
        if self._save_task and not self._save_task.done():
            self._save_task.cancel()

        self._save_task = self.page.run_task(self._save_async)

    async def _save_async(self):
        if self._mask_path is not None and self._mask_data is not None:
                await asyncio.to_thread(np.save, self._mask_path, self._mask_data, allow_pickle=True)

    def delete_mask(self):
        def cancel_dialog(a):
            cupertino_alert_dialog.open = False
            a.control.page.update()

        def ok_dialog(a):

            cupertino_alert_dialog.open = False
            a.control.page.update()
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
                self.update_mask_image()
                self.on_mask_change(self._image_id,True)

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

    def redo_stack(self,e):
        if len(self._redo_stack) == 0:
            return
        self._undo_button.icon_color = ft.Colors.WHITE_60
        self._undo_button.disabled = False
        self._undo_button.update()
        first_list_item = self._redo_stack.pop()

        if first_list_item[0] == "delete_action":
            self._delete_cell(first_list_item[1])
        elif first_list_item[0] == "draw_action":
            self._cell_drawn(first_list_item[1])
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

    def undo_stack(self,e):
        if len(self._undo_stack) == 0:
            return

        self._redo_button.icon_color = ft.Colors.WHITE_60
        self._redo_button.disabled = False
        self._redo_button.update()
        first_list_item = self._undo_stack.pop()
        if first_list_item[0] == "delete_action":
            self._delete_cell(first_list_item[1])
        elif first_list_item[0] == "draw_action":
            self._cell_drawn(first_list_item[1])
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

    def show_ids_and_value(self,pos: tuple):
        if self._mask_path is None or self._mask_button.icon_color == ft.Colors.WHITE_60 or self._mask_button.icon_color == ft.Colors.BLACK_12:
            return

        mask = self._mask_data["masks"]

        if mask.ndim == 3:
            if self._slice_id < 0:
                raise ValueError("slice_id should be non-negative")
            mask = mask[self._slice_id, :, :]

        #if hovered over cell, get cell id
        cell_id = _get_cell_id_from_position(pos, mask)

        if cell_id is None or cell_id == 0:
            self._id_info.visible = False
            self.page.update()
            return

        #load fluorescence value from cache
        cell_value = self._fluorescence_cache.get_fluorescence_value(cell_id,mask,np.array(self._image_cache.get_image(self._main_paths[self._image_id][self._channel_id])),self._channel_id, self._slice_id)
        #cell_value = self._fluorescence_cache.get_fluorescence_value(cell_id, mask, np.array(
         #   self._image_cache.get_image(r"C:\Users\Jenna\Studium\FS5\data\data\output\Series003c2.tif")),self._channel_id, self._slice_id)

        #show id and value in canvas
        if self._show_id_checkbox.selected :
            self._id_info.content.value = (
                f"Cell ID: {cell_id}\n"
                f"Value: {cell_value:.2f}"
            )
            self._id_info.visible = True
            self.page.update()
