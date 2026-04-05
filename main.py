import flet as ft

from image_editing_view import ImageEditingView


def main(page: ft.Page):
    page.title = "CellSePiDrawing Test"
    image_editing_view = ImageEditingView()
    page.add(
        image_editing_view
    )
    image_editing_view._load_main_image_with_path("/home/mmdark/Downloads/data (4)/data/output/Series003c4.tif") #select a image
    image_editing_view._update_mask_image()

ft.run(main)
