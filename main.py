from pathlib import Path

import flet as ft

from image_editing_view import ImageEditingView


def main(page: ft.Page):
    page.title = "CellSePiDrawing Test"
    image_editing_view = ImageEditingView()
    page.add(
        image_editing_view
    )
    image_editing_view._load_main_image_with_path(Path(r"C:\Users\Jenna\Studium\FS5\data\data\output\Series003c2.tif")) #select a image
    image_editing_view.update_mask_image()
ft.run(main)
