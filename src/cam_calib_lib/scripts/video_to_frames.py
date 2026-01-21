from dataclasses import dataclass
from pathlib import Path

import click
import cv2


@dataclass
class MyContext:
    input_video: Path
    output_dir: Path
    target_fps: int = -1


def video_to_frames(ctx: MyContext, max_res: int = 0):
    """Taken from video depth anything"""

    cap = cv2.VideoCapture(str(ctx.input_video))

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if max_res > 0 and max(original_height, original_width) > max_res:
        scale = max_res / max(original_height, original_width)
        height = round(original_height * scale)
        width = round(original_width * scale)

    fps = original_fps if ctx.target_fps < 0 else ctx.target_fps
    stride = max(round(original_fps / fps), 1)

    print(f"Original FPS: {original_fps}, Target FPS: {fps}, Stride: {stride}")

    # frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret: # If no frame is returned exit.
            break

        if frame_count % stride == 0:
            if max_res > 0 and max(original_height, original_width) > max_res:
                frame = cv2.resize(frame, (width, height))  # Resize frame

            cv2.imwrite(
                str(ctx.output_dir / f"frame_{frame_count:06d}.png"),
                frame,
            )


        frame_count += 1
        # print("debug frame_count", frame_count)

    print(f"Extracted {frame_count} frames from {ctx.input_video}.")
    cap.release()


@click.group()
@click.option("--input-video", type=click.Path(), required=True)
@click.option("--output-dir", type=click.Path())
@click.option("--target-fps", type=int, default=-1, help="Target frames per second.")
@click.pass_context
def cli(ctx: click.Context, input_video: str, output_dir: str, target_fps: int):

    if output_dir is None:
        output_dir = Path(input_video).parent / "frames"

    ctx.obj = MyContext(input_video=Path(input_video), output_dir=Path(output_dir), target_fps=target_fps)


# Step 3: Subcommand that uses the context with type hints
@cli.command()
@click.pass_obj
def run(ctx: MyContext):
    print(f"Loading video {ctx.input_video}...")

    if not ctx.input_video.exists():
        raise Exception(f"Error: {ctx.input_video} does not exist.")

    if ctx.output_dir is None:
        ctx.output_dir = ctx.input_video.parent / "frames"
        ctx.output_dir.mkdir(exist_ok=True)
    else:
        ctx.output_dir = Path(ctx.output_dir)
        ctx.output_dir.mkdir(exist_ok=True, parents=True)

    video_to_frames(ctx)

@cli.command()
@click.pass_obj
def show_fps(ctx: MyContext):
    """Show the frames per second of the video."""
    cap = cv2.VideoCapture(str(ctx.input_video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Frames per second of {ctx.input_video}: {fps}")
    cap.release()


# Entry point
if __name__ == "__main__":
    """
    run with

    python video_to_frames.py --input-video . --output-dir ../../ run
    python video_to_frames.py --input-video . --output-dir ../../ show-fps 

    see: https://click.palletsprojects.com/en/stable/api/#context
    """
    cli()
