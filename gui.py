# drawing_gui.py

import io
import os
import numpy as np
import PySimpleGUI as sg
import shutil
import tempfile
import gui_util as util
from PIL import Image, ImageColor, ImageDraw

file_types = [("JPEG (*.jpg)", "*.jpg"), ("All files (*.*)", "*.*")]
tmp_file = tempfile.NamedTemporaryFile(suffix=".jpg").name
default_colors = list(ImageColor.colormap.keys())

def rgb2hex(r, g, b):
    return "#{:02x}{:02x}{:02x}".format(r,g,b)

def get_value(key, values):
    value = values[key]
    if value.isdigit():
        return int(value)
    return 0

def Seg_to_Image(seg_np):
    return Image.fromarray(np.uint8(util.convert_to_onehot(seg_np) * 255))

def load_image(values, window):
    image_file = values["-FILENAME-"]
    seg_file = image_file.replace(image_file[-3:],'npy')
    graph = window["-GRAPH-"]  
    seg_graph = window["-LABEL-"]

    if os.path.exists(image_file):
        shutil.copy(image_file, tmp_file)
        image = Image.open(tmp_file)
        image_original = image.copy()
        image.thumbnail((768, 512))
        image.save(tmp_file)
        width, height = image.size
        bio = io.BytesIO()
        image.save(bio, format="PNG")
        # window["-IMAGE-"].update(data=bio.getvalue())
        graph.draw_image(data=bio.getvalue(), location=(0, 800))
        graph.set_size((width, height))
        graph.update()
    else:
        bio = None
        image_original = None

    if os.path.exists(seg_file):
        seg = np.load(seg_file)
        seg = seg * (seg>-1)
        seg_image = Seg_to_Image(seg)
        seg_image.thumbnail((768, 512))        
        seg_bio = io.BytesIO()
        seg_image.save(seg_bio, format="PNG")
        width, height = seg_image.size
        seg_graph.draw_image(data=seg_bio.getvalue(), location=(0, 800))
        seg_graph.set_size((width, height))
        seg_graph.update()
    else:
        seg = None

    return image_original, seg, bio


def draw_at(window, location, image):
    graph = window["-GRAPH-"]  
    image = Image.fromarray(np.uint8(image))
    bio = io.BytesIO()
    image.save(bio, format="PNG")
    if location is None:
        graph.draw_image(data=bio.getvalue(), location=(0, 800))
    else:
        graph.draw_image(data=bio.getvalue(), location=location)
    graph.update()

def reload_image(window, image, seg):
    graph = window["-GRAPH-"]  
    seg_graph = window["-LABEL-"]
    image = Image.fromarray(np.uint8(image))
    image_original = image.copy()
    image.thumbnail((768, 512))
    width, height = image.size
    bio = io.BytesIO()
    image.save(bio, format="PNG")
    graph.draw_image(data=bio.getvalue(), location=(0, 800))
    graph.set_size((width, height))
    graph.update()
    seg_image = Seg_to_Image(seg)
    seg_image.thumbnail((768, 512))        
    seg_bio = io.BytesIO()
    seg_image.save(seg_bio, format="PNG")
    width, height = seg_image.size
    seg_graph.draw_image(data=seg_bio.getvalue(), location=(0, 800))
    seg_graph.set_size((width, height))
    seg_graph.update()
    return image_original, seg, bio


def refresh_graph(graph, bio):
    graph.draw_image(data=bio.getvalue(), location=(0, 800))
    graph.update()

# def apply_drawing(values, window):
#     image_file = values["-FILENAME-"]
#     graph = window["-GRAPH-"]  
#     shape = values["-SHAPES-"]
#     begin_x = get_value("-BEGIN_X-", values)
#     begin_y = get_value("-BEGIN_Y-", values)
#     end_x = get_value("-END_X-", values)
#     end_y = get_value("-END_Y-", values)
#     width = 2 # get_value("-WIDTH-", values)
#     fill_color = default_colors[0] # values["-FILL_COLOR-"]
#     outline_color = default_colors[2] # values["-OUTLINE_COLOR-"]

#     if os.path.exists(image_file):
#         shutil.copy(image_file, tmp_file)
#         image = Image.open(tmp_file)
#         image.thumbnail((400, 400))
#         draw = ImageDraw.Draw(image)
#         if shape == "Ellipse":
#             draw.ellipse(
#                 (begin_x, begin_y, end_x, end_y),
#                 fill=fill_color,
#                 width=width,
#                 outline=outline_color,
#             )
#         elif shape == "Rectangle":
#             draw.rectangle(
#                 (begin_x, begin_y, end_x, end_y),
#                 fill=fill_color,
#                 width=width,
#                 outline=outline_color,
#             )
#         image.save(tmp_file)
#         bio = io.BytesIO()
#         image.save(bio, format="PNG")
#         graph.draw_image(data=bio.getvalue(), location=(0, 800))
#         graph.set_size((width, height))
#         graph.update()



def create_coords_elements(label, begin_x, begin_y, key1, key2):
    return [
        sg.Text(label),
        sg.Input(begin_x, size=(5, 1), key=key1, enable_events=True),
        sg.Input(begin_y, size=(5, 1), key=key2, enable_events=True),
    ]


def save_image(values):
    save_filename = sg.popup_get_file(
        "File", file_types=file_types, save_as=True, no_window=True
    )
    if save_filename == values["-FILENAME-"]:
        sg.popup_error(
            "You are not allowed to overwrite the original image!")
    else:
        if save_filename:
            shutil.copy(tmp_file, save_filename)
            sg.popup(f"Saved: {save_filename}")


def main():
    colors = list(ImageColor.colormap.keys())

    label_color_map = {
    "Grounds ('R')": rgb2hex(255, 0, 0), 
    "Building ('G')": rgb2hex(0, 255, 0), 
    "Plants ('B')": rgb2hex(0, 0, 255)
    }


    label_int_map = {
    "Grounds ('R')":  0, 
    "Building ('G')":  1, 
    "Plants ('B')": 2
    }

    labels = ["Grounds ('R')", "Building ('G')", "Plants ('B')"]
    load_events = "Load Image"
    clear_events = "Clear Selection"
    click_events = "-GRAPH-"
    process_events = "Process"
    network_events = "Run Inference"

    layout = [
        [
            sg.Text("Image File"),
            sg.Input(
                size=(25, 1), key="-FILENAME-"
            ),
            sg.FileBrowse(file_types=file_types),
            sg.Button(load_events),
        ],
        # [
        #     sg.Text("Shapes"),
        #     sg.Combo(
        #         ["Ellipse", "Rectangle"],
        #         default_value="Rectangle",
        #         key="-SHAPES-",
        #         enable_events=True,
        #         readonly=True,
        #     ),
        # ],
        # [
        #     *create_coords_elements(
        #         "Begin Coords", "10", "10", "-BEGIN_X-", "-BEGIN_Y-"
        #     ),
        #     *create_coords_elements(
        #         "End Coords", "100", "100", "-END_X-", "-END_Y-"
        #     ),
        # ],
        [
            sg.Text("Label Filter"),
            sg.Combo(
                labels,
                default_value=labels[0],
                key="-FILL_COLOR-",
                enable_events=True,
                readonly=True
            ),
            sg.Text("Replaced Label"),
            sg.Combo(
                labels,
                default_value=labels[1],
                key="-OUTLINE_COLOR-",
                enable_events=True,
                readonly=True
            ),
        ],
        [
            sg.Radio("Inpainting Mode", 'group',default=True, k='INPAINTING'),
            sg.Radio("Expansion Mode", 'group', k='EXPANSION'),
        ],

        [
            sg.Graph(
                canvas_size=(400, 400),
                graph_bottom_left=(0, 0),
                graph_top_right=(800, 800),
                key="-GRAPH-",
                enable_events=True,
                background_color='lightblue',
                drag_submits=True,
                right_click_menu=[[],['Erase item',]]
            ), 
            sg.Graph(
                canvas_size=(400, 400),
                graph_bottom_left=(0, 0),
                graph_top_right=(800, 800),
                key="-LABEL-",
                enable_events=True,
                background_color='green',
                right_click_menu=[[],['Erase item',]]
            ), 
        ],
        [sg.Text(key='-INFO-', size=(60, 1))],
        [sg.Button(clear_events)],
        [sg.Button(process_events), sg.Button(network_events)],
    ]

    window = sg.Window("Drawing GUI", layout, size=(1800, 1000))

    graph = window["-GRAPH-"]  
    dragging = False
    start_point = end_point = prior_rect = selected_start = selected_end = None

    current_mask = None
    bio = None
    current_model, current_config = util.initialize()

    events = [
        # "Load Image",
        "-BEGIN_X-",
        "-BEGIN_Y-",
        "-END_X-",
        "-END_Y-",
        "-FILL_COLOR-",
        "-OUTLINE_COLOR-",
        "-WIDTH-",
        "-SHAPES-",
    ]
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        EXPANSION_MODE = values['EXPANSION']

        # load
        if event in [load_events, clear_events]:
            image, seg, bio = load_image(values, window)
            original_image = image

        # process
        if event in process_events:
            if None in (selected_start, selected_end):
                window["-INFO-"].update(value="Please first select a region to remove!")
            else:
                sx, sy = selected_start
                ex, ey = selected_end
                coords = [x / 800 for x in [sx,sy,ex,ey]]
                new_img, new_seg, current_mask = util.process_image(
                    coords, 
                    image,
                    seg,
                    label_int_map[values["-FILL_COLOR-"]], 
                    label_int_map[values["-OUTLINE_COLOR-"]],
                    expansion = EXPANSION_MODE
                )
                image, seg, bio = reload_image(window, new_img, new_seg)

        # infer
        if event in network_events:
            if current_mask is None and not EXPANSION_MODE:
                window["-INFO-"].update(value=f"Please first use the {process_events} function!")
            else:
                if EXPANSION_MODE:
                    data = [new_img, new_seg]
                else:
                    # small trick: modify coords so the inference machine sees more region
                    # sx = max(0, min(sx, ex) - 32)
                    # ex = max(sx, ex)
                    # sy = min(800, max(sy, ey) + 32)
                    # ey = min(sy, ey)
                    # coords = [x / 800 for x in [sx,sy,ex,ey]]                 
                    data = [coords, new_img, new_seg, current_mask]
                current_model, current_config, output, new_loc = util.run(current_model, 
                                                                          current_config, 
                                                                          data,
                                                                          window=window, 
                                                                          expansion=EXPANSION_MODE)
                draw_at(window, new_loc, output)
                # draw_at(window, selected_start, output)
        # click
        if event == click_events:
            x, y = values["-GRAPH-"]
            label_color = label_color_map[values["-FILL_COLOR-"]]
            if bio is not None and not dragging:
                start_point = (x, y)
                refresh_graph(graph, bio)
                dragging = True
            else:
                end_point = (x, y)
            if prior_rect:
                graph.delete_figure(prior_rect)
            if None not in (start_point, end_point):
                prior_rect = graph.draw_rectangle(start_point, end_point, fill_color=label_color, line_width=0)
            window["-INFO-"].update(value=f"mouse {values['-GRAPH-']}")
        elif event.endswith('+UP'):  # The drawing has ended because mouse up
            window["-INFO-"].update(value=f"grabbed rectangle from {start_point} to {end_point}")
            selected_start = start_point
            selected_end = end_point
            start_point, end_point = None, None  # enable grabbing a new rect
            dragging = False
            prior_rect = None
        elif event.endswith('+RIGHT+'):  # Righ click
            window["-INFO-"].update(value=f"Right clicked location {values['-GRAPH-']}")
        elif event.endswith('+MOTION+'):  # Righ click
            window["-INFO-"].update(value=f"mouse freely moving {values['-GRAPH-']}")

        # if event in events:
        #     apply_drawing(values, window)
        # if event == "Save" and values["-FILENAME-"]:
        #     save_image(values)

    window.close()


if __name__ == "__main__":
    main()