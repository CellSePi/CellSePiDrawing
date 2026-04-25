import typing
from dataclasses import dataclass, field

import flet as ft
import flet.canvas as cv

from drawing_util import rgb_to_hex


@dataclass
class State:
    x: float = 0
    y: float = 0
    lines: list[tuple[tuple[float, float],tuple[float, float]]] = field(default_factory=list)
    start_point: tuple[float,float] | None = None
    drawing_mode: bool = False
    delete_mode: bool = False
    cell_info_mode:bool =False
    image_width: int = 0
    image_height: int = 0
    width: float = 0
    height: float = 0
    scale: float = 1
    offset_x: float = 0
    offset_y: float = 0

    def update_scale_offset(self):
        if self.image_width > 0 and self.image_height > 0:
            self.scale = min(self.width/self.image_width, self.height/self.image_height)
            self.offset_x = (self.width - (self.image_width * self.scale))/2
            self.offset_y = (self.height - (self.image_height * self.scale))/2
    
class DrawingTool(cv.Canvas):
    def __init__(self, on_cell_drawn: typing.Callable[[list], None] = None, on_cell_deleted: typing.Callable[[tuple], None] = None,on_show_ids: typing.Callable[[tuple], None] = None):
        super().__init__()
        self.draw_color = rgb_to_hex((0,255,0))
        self.left=0
        self.right=0
        self.top=0
        self.bottom=0
        self.expand = False
        self.on_cell_drawn = on_cell_drawn
        self.on_cell_deleted = on_cell_deleted
        self.show_ids =on_show_ids
        self._state = State()
        self.content = ft.GestureDetector(
            on_tap_down= self.handle_click,
            on_pan_start=self.handle_pan_start,
            on_pan_update=self.handle_pan_update,
            on_pan_end=self.handle_pan_end,
            on_hover = self.handle_hover,
            drag_interval=0,
        )
        self.on_resize=self.on_canvas_resize

    def on_canvas_resize(self, e: cv.CanvasResizeEvent):
        self._state.width = e.width
        self._state.height = e.height
        self._state.update_scale_offset()

    def set_bounds(self,width,height):
        self._state.image_width = width
        self._state.image_height = height
        self._state.update_scale_offset()

    def get_bounds(self):
        return self._state.image_width, self._state.image_height

    def draw(self):
        self._state.drawing_mode = True
        self._state.delete_mode = False

    def deactivate_drawing(self):
        self._state.drawing_mode = False

    def delete(self):
        self._state.delete_mode = True
        self._state.drawing_mode = False

    def deactivate_delete(self):
        self._state.delete_mode = False

    def show_cell_info(self):
        self._state.cell_info_mode = True

    def deactivate_cell_info(self):
        self._state.cell_info_mode = False


    def handle_click(self, e: ft.TapEvent):
        if self._state.delete_mode:
            self.on_cell_deleted(self.translate_into_image_coordinates((e.local_position.x, e.local_position.y)))

    def handle_hover (self, e:ft.HoverEvent):

        if self._state.cell_info_mode:
            self.show_ids(self.translate_into_image_coordinates((e.local_position.x, e.local_position.y)))

    def handle_pan_start(self, e: ft.DragStartEvent):
        if self._state.drawing_mode:
            x, y = self.clamp_to_image_bounds((e.local_position.x, e.local_position.y))
            self._state.x = x
            self._state.y = y
            if self._state.start_point is None:
                self._state.start_point = (x, y)


    async def handle_pan_update(self, e: ft.DragUpdateEvent):
        if self._state.drawing_mode:
            ft.context.disable_auto_update()
            x,y = self.clamp_to_image_bounds((e.local_position.x, e.local_position.y))
            self.shapes.append(
                cv.Line(
                    x1=self._state.x,
                    y1=self._state.y,
                    x2=x,
                    y2=y,
                    paint=ft.Paint(stroke_width=3,color=self.draw_color),
                )
            )
            self._state.lines.append((self.translate_into_image_coordinates((self._state.x, self._state.y)), self.translate_into_image_coordinates((x, y))))
            self.update()

            self._state.x = x
            self._state.y = y

    def handle_pan_end(self):
        if self._state.drawing_mode and self._state.start_point is not None:
            self._state.lines.append((self.translate_into_image_coordinates((self._state.x, self._state.y)), self.translate_into_image_coordinates((self._state.start_point[0], self._state.start_point[1]))))
            self.on_cell_drawn(self._state.lines)
            self._state.lines.clear()
            self.shapes.clear()
            self.update()

        self._state.start_point = None

    def clamp_to_image_bounds(self, point):
        """
        If mouse goes out of image, the line stays in the image bounds.
        """
        max_width = self._state.offset_x + (self._state.image_width * self._state.scale)
        max_height = self._state.offset_y + (self._state.image_height * self._state.scale)

        x = max(self._state.offset_x, min(point[0], max_width - 1))
        y = max(self._state.offset_y, min(point[1], max_height - 1))

        return x, y

    def translate_into_image_coordinates(self, point):
        """
        Translate a point from the drawing coordinates to the image coordinates.
        """
        real_x = (point[0] - self._state.offset_x) / self._state.scale
        real_y = (point[1] - self._state.offset_y) / self._state.scale

        safe_x = int(max(0, min(real_x, self._state.image_width - 1)))
        safe_y = int(max(0, min(real_y, self._state.image_height - 1)))

        return safe_x, safe_y
