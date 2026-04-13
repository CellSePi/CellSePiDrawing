import base64
import os
import typing
from io import BytesIO
from pathlib import Path

import cv2
import flet as ft
import numpy as np
import tifffile
from PIL import Image

from drawing_tool import DrawingTool
from drawing_util import bresenham_line, search_free_id, trace_contour, fill_polygon_from_outline, find_border_pixels, \
    mask_shifting, rgb_to_hex


def load_image(image_path,get_slice=-1):
    image = tifffile.imread(image_path)
    shape = list(image.shape)
    check = image.ndim == 3
    if check:
        if not get_slice == -1:
            image = np.take(image, get_slice, axis=2)
        else:
            image = np.max(image, axis=2)

    _, buffer = cv2.imencode('.png', image)

    return base64.b64encode(buffer).decode('utf-8'),shape,check

def convert_npy_to_canvas(mask, outline, mask_color, outline_color, opacity, slice_id=-1):
    """
    handles the conversion of the given file data

    Args:
        mask= the mask data stored in the numpy directory
        outline= the outline data stored in the numpy directory
    """
    buffer= BytesIO()

    if mask.ndim == 3:
        if slice_id >= 0:
            mask = np.take(mask, slice_id, axis=0)
        else:
            mask = np.max(mask, axis=0)

    if outline.ndim == 3:
        if slice_id >= 0:
            outline = np.take(outline, slice_id, axis=0)
        else:
            outline = np.max(outline, axis=0)

    image_mask = np.zeros(shape=(mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    r,g,b = mask_color
    image_mask[mask != 0] = (r, g, b, opacity)
    r, g, b = outline_color
    image_mask[outline != 0] = (r, g, b, 255)
    im= Image.fromarray(image_mask).convert("RGBA")

    #saves the image as a image(base64)
    im.save(buffer, format="PNG")
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

class ImageEditingView(ft.Card):
    def __init__(self,on_mask_change: typing.Callable[[str], None] = None):
        super().__init__()
        self._mask_paths = None
        self._main_paths = None
        self._mask_path = None #Could set a mask_path for TESTING
        self._slice_id = -1
        self._image_3d = False
        self._image_id = None
        self._channel_id = None
        self._seg_channel_id = None
        self.mask_color = (255, 0, 0)
        self.outline_color = (0, 255, 0)
        self.mask_opacity = 128
        self._user_2_5d = False
        self.on_mask_change = on_mask_change or (lambda x: None)
        self.mask_suffix = "_seg"
        self.expand=True
        self._edit_allowed = True
        self._mask_image = ft.Image(src="Placeholder", fit=ft.BoxFit.CONTAIN, visible=False,gapless_playback=True,expand=True)
        self._main_image = ft.Image(src="Placeholder", fit=ft.BoxFit.CONTAIN,visible=False,gapless_playback=True,expand=True)
        self.drawing_tool = DrawingTool(on_cell_drawn=self._cell_drawn, on_cell_deleted=self._delete_cell)
        self.image_stack = ft.InteractiveViewer(content=ft.Stack([self._main_image, self._mask_image, self.drawing_tool],expand=True),expand=True)
        self._mask_button = ft.IconButton(icon=ft.Icons.REMOVE_RED_EYE, icon_color=ft.Colors.BLACK12,
                                          style=ft.ButtonStyle(
                                              shape=ft.RoundedRectangleBorder(radius=12),), on_click=lambda e: self._show_mask(),
                                          tooltip="Show Mask", hover_color=ft.Colors.WHITE12, disabled=False)
        self._edit_button = ft.IconButton(icon=ft.Icons.BRUSH, icon_color=ft.Colors.BLACK_12,
                                          style=ft.ButtonStyle(
                                              shape=ft.RoundedRectangleBorder(radius=12), ),disabled=True,
                                          tooltip="Draw Mode", hover_color=ft.Colors.WHITE_12,on_click=lambda e:self._toggle_draw())
        self._delete_button = ft.IconButton(icon=ft.Icons.CLEAR, icon_color=ft.Colors.WHITE_60,
                                          style=ft.ButtonStyle(
                                              shape=ft.RoundedRectangleBorder(radius=12), ),
                                          tooltip="Delete Mode", hover_color=ft.Colors.WHITE12,on_click=lambda e: self._toggle_delete())
        self._delete_mask_button = ft.IconButton(icon=ft.Icons.DELETE_FOREVER, icon_color=ft.Colors.WHITE_60,
                                            style=ft.ButtonStyle(
                                                shape=ft.RoundedRectangleBorder(radius=12), ),
                                            tooltip="Delete the complete mask.", hover_color=ft.Colors.WHITE12,
                                            on_click=lambda e: self.delete_mask())
        self._slider_2_5d = ft.Slider(
            min=0, max=100, divisions=None, label="Slice: {value}",value=0,
            opacity=1.0 if self._user_2_5d else 0.0, height=20,
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
            icon=ft.Icons.EXPAND,
            icon_color=ft.Colors.WHITE60,
            style=ft.ButtonStyle(
                shape=ft.RoundedRectangleBorder(radius=12),),
            hover_color=ft.Colors.WHITE12,
            selected_icon=ft.Icons.COMPRESS_ROUNDED,
            selected_icon_color=ft.Colors.WHITE,
            selected=False,
            on_click=lambda e: self._toggle_shifting(e),
        )
        self.control_tools = ft.Container(ft.Container(ft.Row(
                [   self._shifting_check_box,
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
                ], spacing=2,alignment=ft.MainAxisAlignment.CENTER,
            ), bgcolor=ft.Colors.BLUE_400, expand=True, border_radius=ft.border_radius.vertical(top=0, bottom=12),
            ))
        #TODO: ADD REDO/UNDO
        self.content = ft.Column(controls=[ft.Container(self.image_stack,alignment=ft.Alignment.CENTER,expand=True),self.control_tools],spacing=0)

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
        self._mask_path = None
        self._mask_image.src = "Placeholder"
        self._mask_image.visible = False
        self._mask_image.update()
        self._mask_button.tooltip = "Show mask"
        self._mask_button.icon_color = ft.Colors.BLACK12
        self._mask_button.disabled = True
        self._mask_button.update()

    def select_image(self, img_id, channel_id,seg_channel_id):
        if self._seg_channel_id != seg_channel_id or self._image_id != img_id:
            self._load_mask_image(img_id, seg_channel_id)
        self._image_id = img_id
        self._channel_id = channel_id
        self._seg_channel_id = seg_channel_id
        self._load_main_image(img_id, channel_id)
        #TODO: reset undo/redo when a new image is selected

    def _load_main_image(self, img_id, channel_id):
        if self._main_paths is not None:
            if img_id in self._main_paths:
                if channel_id in self._main_paths[img_id]:
                    src, shape,img_3d = load_image(self._main_paths[img_id][channel_id], get_slice=self._slice_id)
                    self._main_image.src = src
                    self._main_image.visible = True
                    self.drawing_tool.set_bounds(shape[1], shape[0])
                    self._main_image.update()
                    if img_3d:
                        self._image_3d = True
                        if self._slider_2_5d.opacity == 1.0 and self._edit_allowed:
                            self._edit_button.icon_color = ft.Colors.WHITE60
                            self._edit_button.disabled = False
                            self._edit_button.update()
                        else:
                            self._edit_button.icon_color = ft.Colors.BLACK12
                            self._edit_button.disabled = True
                            self._edit_button.update()
                        self._slider_2_5d.value = 0 if shape[-2] - 1 < self._slider_2_5d.value else self._slider_2_5d.value
                        self._slider_2_5d.max = shape[2] - 1
                        self._slider_2_5d.divisions = shape[2] - 2
                        self._slider_2_5d.disabled = False
                        self._slider_2_5d.update()
                    else:
                        self._image_3d = False
                        if self._edit_allowed:
                            self._edit_button.icon_color = ft.Colors.WHITE60
                            self._edit_button.disabled = False
                            self._edit_button.update()
                        self._slider_2_5d.value = 0
                        self._slider_2_5d.max = 1
                        self._slider_2_5d.divisions = None
                        self._slider_2_5d.disabled = True
                        self._slider_2_5d.update()
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
        self._load_main_image(self._image_id,self._channel_id)
        self.update_mask_image()

    def _load_main_image_with_path(self,path):
        #ONLY FOR TESTING TODO:DELETE AFTER IMPLEMENTING IN CELLSEPI
        src, shape, img_3d = load_image(path, get_slice=-1)
        self._main_image.src = src
        self._main_image.visible = True
        self.drawing_tool.set_bounds(shape[1],shape[0])
        self._main_image.update()
        if img_3d:
            self._image_3d = True
            if self._slider_2_5d.opacity == 1.0 and self._edit_allowed:
                self._edit_button.icon_color = ft.Colors.WHITE60
                self._edit_button.disabled = False
                self._edit_button.update()
            else:
                self._edit_button.icon_color = ft.Colors.BLACK12
                self._edit_button.disabled = True
                self._edit_button.update()
            self._slider_2_5d.value = 0 if shape[
                                               -2] - 1 < self._slider_2_5d.value else self._slider_2_5d.value
            self._slider_2_5d.max = shape[2] - 1
            self._slider_2_5d.divisions = shape[2] - 2
            self._slider_2_5d.disabled = False
            self._slider_2_5d.update()
        else:
            self._image_3d = False
            if self._edit_allowed:
                self._edit_button.icon_color = ft.Colors.WHITE60
                self._edit_button.disabled = False
                self._edit_button.update()
            self._slider_2_5d.value = 0
            self._slider_2_5d.max = 1
            self._slider_2_5d.divisions = None
            self._slider_2_5d.disabled = True
            self._slider_2_5d.update()
        return

    def _load_mask_image(self, img_id, seg_channel_id):
        if self._mask_paths is not None:
            if img_id in self._mask_paths:
                if seg_channel_id in self._mask_paths[img_id]:
                    mask_data = np.load(
                        Path(self._mask_paths[img_id][seg_channel_id]),allow_pickle=True).item()
                    self._mask_path = self._mask_paths[img_id][seg_channel_id]
                    mask = mask_data["masks"]
                    outline = mask_data["outlines"]
                    self._mask_image.src = convert_npy_to_canvas(mask, outline, self.mask_color, self.outline_color, self.mask_opacity, slice_id=self._slice_id)
                    self._mask_image.update()
                    if not self._mask_image.visible:
                        self._mask_button.icon_color = ft.Colors.WHITE60
                        self._mask_button.tooltip = "Show mask"
                        self._mask_button.disabled = False
                        self._mask_button.update()
                    return

        self._mask_path = None
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
        elif self._mask_paths is not None and self._image_id in self._mask_paths and self._channel_id in self._mask_paths[self._image_id] and self._mask_paths[self._image_id][self._channel_id] is not None:
            self._mask_path = self._mask_paths[self._image_id][self._channel_id]
            self._update_mask_image()
        else:
            self._mask_image.src = "Placeholder"
            self._mask_image.visible = False
            self._mask_image.update()
            self._mask_button.tooltip = "Show mask"
            self._mask_button.icon_color = ft.Colors.BLACK12
            self._mask_button.disabled = True
            self._mask_button.update()

    def _update_mask_image(self):
        if not self._mask_image.visible:
            self._mask_button.icon_color = ft.Colors.WHITE60
            self._mask_button.tooltip = "Show mask"
            self._mask_button.disabled = False
            self._mask_button.update()
        mask_data = np.load(
            Path(self._mask_path), allow_pickle=True).item()
        mask = mask_data["masks"]
        outline = mask_data["outlines"]
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

    def _cell_drawn(self, lines_data: list):
        #update the mask data
        # gets the pixels that build the lines of the drawn cell
        if self._mask_path is None: #currently no mask is given
            if self._image_id is None or self._seg_channel_id is None:
                return
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
                empty_mask = {
                    "masks": np.zeros((image_height, image_width), dtype=np.uint8),
                    "outlines": np.zeros((image_height, image_width), dtype=np.uint8)
                }
            else:
                #3D-Image Case (with Z-Slices)
                empty_mask = {
                    "masks": np.zeros((self._slider_2_5d.max + 1, image_height, image_width), dtype=np.uint8),
                    "outlines": np.zeros((self._slider_2_5d.max + 1, image_height, image_width), dtype=np.uint8)
                }
            #Save the new empty mask
            np.save(self._mask_path, empty_mask)


        line_pixels = set()
        for line in lines_data:
            pixels = bresenham_line(line[0], line[1])  # Calculates the pixels along the line
            line_pixels.update(pixels)

        mask_data = np.load(self._mask_path, allow_pickle=True).item()
        mask = mask_data["masks"]
        outline = mask_data["outlines"]
        if mask.ndim == 3:
            if self._slice_id < 0:
                raise ValueError("slice_id should be non-negative")
            mask = np.take(mask, self._slice_id, axis=0)

        if outline.ndim == 3:
            if self._slice_id < 0:
                raise ValueError("slice_id should be non-negative")
            outline = np.take(outline, self._slice_id, axis=0)

        free_id = search_free_id(mask, outline)  # search for the next free id in mask and outline

        # add the outline of the new mask (only the parts which not overlap with already existing cells) to outline npy array and fill the complete outline to new_cell_outline to calculate inner pixels
        new_cell_outline = np.zeros_like(outline, dtype=np.uint8)
        for x, y in line_pixels:
            if 0 <= x < outline.shape[1] and 0 <= y < outline.shape[0]:
                new_cell_outline[y, x] = 1
                if outline[y, x] == 0 and mask[y, x] == 0:
                    outline[y, x] = free_id

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
            mask_3d = mask_data["masks"]
            outline_3d = mask_data["outlines"]

            if mask_3d.ndim == 3:
                mask_3d[self._slice_id, :, :] = mask

            if outline_3d.ndim == 3:
                outline_3d[self._slice_id, :, :] = outline

        np.save(self._mask_path, {"masks": mask if self._slice_id == -1 else mask_3d,
                            "outlines": outline if self._slice_id == -1 else outline_3d}, allow_pickle=True)

        self.update_mask_image()
        self.on_mask_change(self._image_id)

    def _delete_cell(self, pos: tuple):
        #delete the cell in the mask data
        if self._mask_path is None:
            return

        mask_data = np.load(self._mask_path, allow_pickle=True).item()

        mask = mask_data["masks"]
        outline = mask_data["outlines"]

        if mask.ndim == 3:
            if self._slice_id < 0:
                raise ValueError("slice_id should be non-negative")
            mask = np.take(mask, self._slice_id, axis=0)

        if outline.ndim == 3:
            if self._slice_id < 0:
                raise ValueError("slice_id should be non-negative")
            outline = np.take(outline, self._slice_id, axis=0)

        cell_id = _get_cell_id_from_position(pos, mask)

        if not cell_id:
            cell_id_outline = _get_cell_id_from_position(pos, outline)
            if not cell_id_outline:
                return
            cell_id = cell_id_outline

        # Update the mask and outline (delete the cell)
        cell_mask = (mask == cell_id).copy()
        cell_outline = (outline == cell_id).copy()
        mask[cell_mask] = 0
        outline[cell_outline] = 0
        if self._shifting_check_box.selected:
            mask_shifting(mask_data, cell_id, self._slice_id)

        mask_3d = None
        outline_3d = None
        if self._slice_id >= 0:
            mask_3d = mask_data["masks"]
            outline_3d = mask_data["outlines"]

            if mask_3d.ndim == 3:
                mask_3d[self._slice_id, :, :] = mask

            if outline_3d.ndim == 3:
                outline_3d[self._slice_id, :, :] = outline

        final_masks = mask if self._slice_id == -1 else mask_3d
        final_outlines = outline if self._slice_id == -1 else outline_3d

        np.save(self._mask_path, {"masks": final_masks,
                                  "outlines": final_outlines}, allow_pickle=True)

        self.update_mask_image()
        self.on_mask_change(self._image_id)

    def delete_mask(self):
        def cancel_dialog(a):
            cupertino_alert_dialog.open = False
            a.control.page.update()

        def ok_dialog(a):
            #TODO: RESET here the undo redo operations
            cupertino_alert_dialog.open = False
            a.control.page.update()
            if self._mask_path is not None:
                if os.path.exists(self._mask_path):
                    os.remove(self._mask_path)
                if self._mask_paths and self._image_id in self._mask_paths:
                    self._mask_paths[self._image_id].pop(self._seg_channel_id, None)
                self._mask_path = None
                self.update_mask_image()
                self.on_mask_change(self._image_id)

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
