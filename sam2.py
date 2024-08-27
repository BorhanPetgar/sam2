"""
INSTALLATION: python3 -m pip install 'git+https://github.com/facebookresearch/segment-anything-2.git'
YOU MUST HAVE A FOLDER CONTAINING THE FRAME IMAGES OF THE VIDEO!
YOU CAN DO THAT BY RUNNING THE FOLLOWING COMMAND:
!ffmpeg -i </path/to/video.mp4> -q:v 2 -start_number 0  /path/to/output_dir/'%05d.jpg'

"""


import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
from PIL import Image

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )
    
from sam2.build_sam import build_sam2_video_predictor
    
def show_mask(mask, frame, obj_id=None, random_color=False):
    if random_color:
        color = np.random.random(3) * 255
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array(cmap(cmap_idx)[:3]) * 255

    mask = mask.astype(bool)
    for c in range(3):  # Apply the mask to each channel
        frame[:, :, c] = np.where(mask, frame[:, :, c] * 0.4 + color[c] * 0.6, frame[:, :, c])

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    
def extract_frames(input_video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    frame_idx = 0

    # Iterate through the video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save the frame as an image
        output_frame_path = os.path.join(output_dir, f"{frame_idx:05d}.jpg")
        cv2.imwrite(output_frame_path, frame)
        frame_idx += 1

    # Release the video capture object
    cap.release()

    print(f"Extracted {frame_idx} frames to {output_dir}")
    
def create_video(frame_files, output_video_path, output_frames_dir):
    if not frame_files:
        print("No frames found to create the video.")
    else:
        first_frame = cv2.imread(os.path.join(output_frames_dir, frame_files[0]))
        height, width, layers = first_frame.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))

        for frame_file in frame_files:
            frame_path = os.path.join(output_frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            video.write(frame)

        video.release()
        print(f"Video saved to {output_video_path}")

def visualize_results(frame_names, video_segments, video_dir, segmented_frames, saved_video_path):
    # Render the segmentation results every few frames and save them as images
    vis_frame_stride = 1
    output_frames_dir = segmented_frames
    os.makedirs(output_frames_dir, exist_ok=True)

    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
        frame_path = os.path.join(video_dir, frame_names[out_frame_idx])
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB for consistency with PIL

        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask(out_mask, frame, obj_id=out_obj_id)

        output_frame_path = os.path.join(output_frames_dir, f"{out_frame_idx}.png")
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert back to BGR for saving with OpenCV
        cv2.imwrite(output_frame_path, frame)
    
    # Create a video from the saved frames
    output_video_path = saved_video_path
    frame_files = sorted([f for f in os.listdir(output_frames_dir) if f.endswith('.png')],
                        key=lambda x: int(os.path.splitext(x)[0]))
    return frame_files, output_video_path, output_frames_dir 

def inference(model_cfg, sam2_checkpoint, video_dir, points, labels=None, prompt_is_point=True, box=None):
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    # scan all the JPEG frame names in this directory
    frame_names = [
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    inference_state = predictor.init_state(video_path=video_dir)

    predictor.reset_state(inference_state)

    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
    
    if prompt_is_point:
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
    )
    else:
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            box=box,
        )

    # Assuming predictor and inference_state are already defined
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    return frame_names, video_segments

def run(input_video, video_dir, sam2_checkpoint, model_cfg, segmented_frames, saved_video_path, points, labels, prompt_is_point, box=None):
    
    extract_frames(input_video_path=input_video, output_dir=video_dir)

    frame_names, video_segments = inference(model_cfg, sam2_checkpoint, video_dir, points, labels, prompt_is_point, box)

    frame_files, output_video_path, output_frames_dir = visualize_results(frame_names, video_segments, video_dir, segmented_frames, saved_video_path)

    create_video(frame_files, output_video_path, output_frames_dir)


if __name__ == "__main__":
    # Enter the path to the video
    input_video = '/path/to/video.mp4'
    # Enter the path the frames of the video will be saved
    video_dir = 'path/to/dir'

    # Enter the path which the segmented frames will be saved
    segmented_frames = "path/to/dir"
    # Enter the path which the segmented video will be saved
    saved_video_path = "/path/to/video.mp4"

    # Enter the path to the SAM2 checkpoint and the model config file
    sam2_checkpoint = "/path/to/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    
    ########################### NOTES ######################################
    """
    *** POINTS ***
    for multiple points, you can add them as a 2D array:
    points = np.array([[x1, y1], [x2, y2], ...], dtype=np.float32)

    for multiple labels, you can add them as a 1D array:
    labels = np.array([1, 1, ...], np.int32)
    
    *** BOX ***
    box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
    """
    ########################################################################

    prompt = np.array([306, 360, 350, 410], dtype=np.float32)
    
    if prompt.shape[-1] == 4:
        prompt_is_point = False
        box = prompt
        points = None
        labels = None
    elif prompt.shape[-1] == 2:
        prompt_is_point = True
        points = prompt
        labels = np.array([1], np.int32) # enter the label(s) for the point(s)
        box = None

    run(input_video, video_dir, sam2_checkpoint, model_cfg, segmented_frames, saved_video_path, points, labels, prompt_is_point, box)
