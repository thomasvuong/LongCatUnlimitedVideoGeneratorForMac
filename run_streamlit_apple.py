import os
import sys
import tempfile
import warnings
from pathlib import Path

import torch
import math
import streamlit as st
import numpy as np
from PIL import Image
import imageio

# Suppress warnings
warnings.filterwarnings("ignore")

# Add the longcat_video module to path
sys.path.append(str(Path(__file__).parent))

from transformers import AutoTokenizer, UMT5EncoderModel
from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
from longcat_video.modules.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel


@st.cache_resource
def load_pipeline():
    """Load the LongCat-Video pipeline (cached for performance)"""
    device = torch.device("cpu")  # Force CPU for Apple Silicon
    checkpoint_dir = str(Path(__file__).parent / "weights" / "LongCat-Video")

    if not Path(checkpoint_dir).exists():
        st.error(f"Model directory not found: {checkpoint_dir}")
        st.stop()

    with st.spinner("Loading LongCat-Video pipeline... This may take a few minutes."):
        try:
            # Load components
            tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_dir, subfolder="tokenizer"
            )

            text_encoder = UMT5EncoderModel.from_pretrained(
                checkpoint_dir,
                subfolder="text_encoder",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            ).to(device)

            vae = AutoencoderKLWan.from_pretrained(
                checkpoint_dir,
                subfolder="vae",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            ).to(device)

            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                checkpoint_dir, subfolder="scheduler"
            )

            transformer = LongCatVideoTransformer3DModel.from_pretrained(
                checkpoint_dir,
                subfolder="dit",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            ).to(device)

            # Create pipeline
            pipeline = LongCatVideoPipeline(
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                dit=transformer,
                scheduler=scheduler,
                vae=vae,
            )

            pipeline.device = "cpu"

            return pipeline

        except Exception as e:
            st.error(f"Error loading pipeline: {str(e)}")
            st.stop()


def save_video(frames, output_path, fps=8):
    """Save video frames to file"""
    try:
        # Handle different frame formats
        frames_array = []

        # Debug: Print shape information
        if isinstance(frames, np.ndarray):
            print(f"DEBUG: frames shape: {frames.shape}, dtype: {frames.dtype}")

        # Check if frames is from pipeline (B, N, H, W, C) or (N, H, W, C)
        if isinstance(frames, np.ndarray):
            if len(frames.shape) == 5:
                # Shape is (B, N, H, W, C) - take first batch
                frames = frames[0]

            if len(frames.shape) == 4:
                # Shape is (N, H, W, C)
                num_frames = frames.shape[0]

                # Convert each frame
                for i in range(num_frames):
                    frame = frames[i]

                    # Ensure frame has correct shape (H, W, C)
                    if len(frame.shape) == 2:
                        # Grayscale - add channel dimension
                        frame = np.expand_dims(frame, axis=-1)

                    # Convert dtype to uint8 if needed
                    if frame.dtype != np.uint8:
                        if frame.max() <= 1.0:
                            frame = (frame * 255).astype(np.uint8)
                        else:
                            frame = np.clip(frame, 0, 255).astype(np.uint8)

                    # Ensure frame has 3 channels for video
                    if frame.shape[-1] == 1:
                        # Convert grayscale to RGB
                        frame = np.repeat(frame, 3, axis=-1)
                    elif frame.shape[-1] == 4:
                        # Convert RGBA to RGB (drop alpha channel)
                        frame = frame[:, :, :3]
                    elif frame.shape[-1] != 3:
                        raise ValueError(
                            f"Unexpected number of channels: {frame.shape[-1]}"
                        )

                    frames_array.append(frame)
            else:
                raise ValueError(f"Unexpected frames shape: {frames.shape}")
        else:
            # Handle list of frames (PIL Images or numpy arrays)
            for frame in frames:
                if isinstance(frame, Image.Image):
                    frame_array = np.array(frame)
                else:
                    frame_array = frame

                if frame_array.dtype != np.uint8:
                    if frame_array.max() <= 1.0:
                        frame_array = (frame_array * 255).astype(np.uint8)
                    else:
                        frame_array = np.clip(frame_array, 0, 255).astype(np.uint8)

                # Ensure 3 channels
                if len(frame_array.shape) == 2:
                    frame_array = np.repeat(np.expand_dims(frame_array, -1), 3, axis=-1)
                elif frame_array.shape[-1] == 1:
                    frame_array = np.repeat(frame_array, 3, axis=-1)
                elif frame_array.shape[-1] == 4:
                    frame_array = frame_array[:, :, :3]

                frames_array.append(frame_array)

        if len(frames_array) == 0:
            raise ValueError("No valid frames to save")

        # Save as MP4
        print(
            f"DEBUG: Saving {len(frames_array)} frames, first frame shape: {frames_array[0].shape}"
        )
        imageio.mimsave(output_path, frames_array, fps=fps, codec="libx264")
        return True
    except Exception as e:
        st.error(f"Error saving video: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def main():
    st.set_page_config(
        page_title="LongCat-Video for Apple Silicon", page_icon="🎬", layout="wide"
    )

    st.title("🍎 LongCat-Video for Apple Silicon")
    st.markdown(
        "Generate high-quality videos from text prompts using LongCat-Video on your Mac!"
    )

    # Sidebar for settings
    st.sidebar.header("⚙️ Generation Settings")

    # Performance warning
    st.sidebar.warning(
        "⚠️ **Performance Note**: Running on CPU (Apple Silicon). "
        "Generation will take 10-60+ minutes depending on settings."
    )

    # Load pipeline
    pipeline = load_pipeline()
    st.sidebar.success("✅ Pipeline loaded successfully!")

    # Main interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📝 Prompt Input")

        # Text inputs
        prompt = st.text_area(
            "Enter your video description:",
            value="A cute cat sitting on a windowsill, looking outside at falling snow",
            height=100,
            help="Describe the video you want to generate in detail",
        )

        negative_prompt = st.text_area(
            "Negative prompt (optional):",
            value="blurry, low quality, distorted, ugly, static",
            height=80,
            help="Describe what you don't want in the video",
        )

        # Generation settings
        st.header("🎛️ Video Settings")

        col1a, col1b = st.columns(2)

        with col1a:
            resolution_options = {
                "256x256 (Fast)": (256, 256),
                "480x480 (Balanced)": (480, 480),
                "720x720 (High Quality)": (720, 720),
            }
            resolution_choice = st.selectbox(
                "Resolution",
                options=list(resolution_options.keys()),
                index=0,
                help="Higher resolution = better quality but much slower",
            )
            width, height = resolution_options[resolution_choice]

            # Ensure frames are compatible (divisible by 4 + 1)
            frame_options = [5, 9, 13, 17, 25, 33]
            num_frames = st.selectbox(
                "Number of Frames (per segment)",
                options=frame_options,
                index=1,  # Default to 9 frames
                help="More frames = longer video but slower generation. If you want a multi-minute video, use Segmented mode below.",
            )

            st.markdown("---")
            st.subheader("Duration / Long-video settings")

            # FPS selection affects final duration calculation
            fps = st.selectbox(
                "Output FPS",
                options=[8, 15, 24, 30],
                index=1,
                help="Frames per second when saving the final video. Higher FPS -> smoother, larger file",
            )

            # Mode: single segment or segmented stitching
            long_mode = st.radio(
                "Generation Mode",
                options=["Single segment", "Segmented (stitch multiple segments)"],
                index=0,
                help="Segmented mode stitches multiple generation calls to create long videos (recommended for minutes-long outputs)",
            )

            # Duration presets (seconds) and custom input
            duration_preset = st.selectbox(
                "Target duration (preset)",
                options=["None", "5s", "8s", "10s", "30s", "1m", "2m", "5m"],
                index=0,
                help="Choose a target duration. In Segmented mode the UI will compute how many segments to generate.",
            )

            custom_seconds = None
            if duration_preset == "None":
                custom_seconds = st.number_input("Custom duration (seconds)", min_value=1, max_value=60*60, value=10)
            else:
                if duration_preset.endswith("s"):
                    custom_seconds = int(duration_preset[:-1])
                elif duration_preset.endswith("m"):
                    custom_seconds = int(duration_preset[:-1]) * 60

            # Segmented settings: per-segment frames and conditioning overlap
            if long_mode.startswith("Segmented"):
                seg_num_frames = st.number_input(
                    "Segment frames (N)",
                    min_value=5,
                    max_value=512,
                    value=93,
                    step=4,
                    help="Number of frames generated per segment (must satisfy (N-1)%vae_scale==0). Larger segments use more memory/time.",
                )

                num_cond_frames = st.number_input(
                    "Conditioning frames (C)",
                    min_value=1,
                    max_value=seg_num_frames - 1,
                    value=13,
                    help="How many frames from previous segment are used as conditioning/overlap. Typical: 13",
                )

                # Compute needed segments to hit the target duration
                desired_total_frames = int(custom_seconds * fps)
                if seg_num_frames <= num_cond_frames:
                    est_segments = 0
                else:
                    if desired_total_frames <= seg_num_frames:
                        est_segments = 0
                    else:
                        est_segments = math.ceil((desired_total_frames - seg_num_frames) / (seg_num_frames - num_cond_frames))

                st.markdown(f"Estimated total frames: {seg_num_frames + est_segments * (seg_num_frames - num_cond_frames)} (≈ {round((seg_num_frames + est_segments * (seg_num_frames - num_cond_frames))/fps,1)}s)")
                st.markdown(f"Segments to generate after initial: {est_segments}")
                generate_segments = st.number_input("Segments to run (0 = auto estimated)", min_value=0, max_value=1000, value=0)
                if generate_segments == 0:
                    generate_segments = est_segments

        with col1b:
            num_inference_steps = st.slider(
                "Inference Steps",
                min_value=5,
                max_value=50,
                value=10,
                step=5,
                help="More steps = better quality but slower",
            )

            guidance_scale = st.slider(
                "Guidance Scale",
                min_value=1.0,
                max_value=10.0,
                value=4.0,
                step=0.5,
                help="Higher values follow prompt more closely",
            )

            seed = st.number_input(
                "Seed (for reproducibility)",
                min_value=-1,
                max_value=999999,
                value=42,
                help="Use -1 for random seed",
            )

        # Estimate generation time
        time_estimate = estimate_generation_time(
            width, height, num_frames, num_inference_steps
        )
        st.info(f"⏱️ Estimated generation time: {time_estimate}")

        # Generate button
            if st.button("🎬 Generate Video", type="primary", use_container_width=True):
                # Pass long-video related settings into the generator
                if long_mode.startswith("Segmented"):
                    generate_video_ui(
                        pipeline,
                        prompt,
                        negative_prompt,
                        width,
                        height,
                        num_frames,
                        num_inference_steps,
                        guidance_scale,
                        seed,
                        long_mode=long_mode,
                        fps=fps,
                        seg_num_frames=seg_num_frames,
                        num_cond_frames=num_cond_frames,
                        generate_segments=generate_segments,
                    )
                else:
                    generate_video_ui(
                        pipeline,
                        prompt,
                        negative_prompt,
                        width,
                        height,
                        num_frames,
                        num_inference_steps,
                        guidance_scale,
                        seed,
                        long_mode=long_mode,
                        fps=fps,
                        seg_num_frames=None,
                        num_cond_frames=None,
                        generate_segments=0,
                    )

    with col2:
        st.header("💡 Tips")
        st.markdown("""
        **For faster generation:**
        - Use 256x256 resolution
        - Use 5-9 frames
        - Use 5-10 inference steps
        - Lower guidance scale (3.0-4.0)

        **For better quality:**
        - Use 480x480 or 720x720
        - Use 17+ frames
        - Use 20+ inference steps
        - Higher guidance scale (6.0-8.0)

        **Apple Silicon Notes:**
        - Uses CPU only (Conv3D not supported on MPS)
        - Requires 32GB+ RAM for optimal performance
        - First generation may take longer due to compilation
        """)

        st.header("📊 System Info")
        st.text(f"Device: CPU (Apple Silicon)")
        st.text(f"PyTorch: {torch.__version__}")

        # Memory info
        import psutil

        memory = psutil.virtual_memory()
        st.text(f"RAM: {memory.total // (1024**3)}GB total")
        st.text(f"Available: {memory.available // (1024**3)}GB")


def estimate_generation_time(width, height, num_frames, num_steps):
    """Estimate generation time based on settings"""
    # Base time for 256x256, 9 frames, 5 steps: ~10 minutes
    base_time = 10

    # Scale by resolution
    resolution_factor = (width * height) / (256 * 256)

    # Scale by frames
    frame_factor = num_frames / 9

    # Scale by steps
    step_factor = num_steps / 5

    total_time = base_time * resolution_factor * frame_factor * step_factor

    if total_time < 5:
        return "5-10 minutes"
    elif total_time < 20:
        return f"{int(total_time)}-{int(total_time * 1.5)} minutes"
    elif total_time < 60:
        return f"{int(total_time // 5) * 5}-{int(total_time * 1.2 // 5) * 5} minutes"
    else:
        return (
            f"{int(total_time // 30) * 30}-{int(total_time * 1.3 // 30) * 30} minutes"
        )


def generate_video_ui(
    pipeline,
    prompt,
    negative_prompt,
    width,
    height,
    num_frames,
    num_inference_steps,
    guidance_scale,
    seed,
    long_mode="Single segment",
    fps=15,
    seg_num_frames=None,
    num_cond_frames=13,
    generate_segments=0,
):
    """Generate video with UI updates"""

    # Create progress containers
    progress_container = st.container()
    result_container = st.container()

    with progress_container:
        st.header("🎥 Generating Video...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.text("Initializing generation...")

            try:
            # Set up generator
            device = torch.device("cpu")
            generator = torch.Generator(device=device).manual_seed(
                seed if seed >= 0 else None
            )

            # Set number of threads for better CPU performance
            torch.set_num_threads(4)

            status_text.text(f"Generating {num_frames} frames at {width}x{height}...")
            progress_bar.progress(10)

            # Generate video
            with torch.inference_mode():
                # Segmented long-video flow
                if long_mode.startswith("Segmented") and seg_num_frames is not None:
                    # Validate temporal constraints: pipeline expects (num_frames - 1) % vae_scale_factor_temporal == 0
                    vae_scale = getattr(pipeline, "vae_scale_factor_temporal", None)
                    try:
                        vae_scale = pipeline.vae.config.scale_factor_temporal
                    except Exception:
                        vae_scale = getattr(pipeline, "vae_scale_factor_temporal", 4)

                    # Adjust seg_num_frames to satisfy constraint
                    if vae_scale and (seg_num_frames - 1) % vae_scale != 0:
                        seg_num_frames = seg_num_frames // vae_scale * vae_scale + 1

                    # Initial segment: T2V to create base video
                    result = pipeline.generate_t2v(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_frames=seg_num_frames,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    )

                    # Extract frames from result
                    if isinstance(result, (list, tuple)):
                        base = result[0]
                    elif hasattr(result, "frames"):
                        base = result.frames[0]
                    else:
                        base = result

                    # Convert to list of frames (PIL-compatible or numpy)
                    if isinstance(base, np.ndarray):
                        if base.ndim == 5:
                            base_frames = [base[0, i] for i in range(base.shape[1])]
                        elif base.ndim == 4:
                            base_frames = [base[i] for i in range(base.shape[0])]
                        else:
                            base_frames = list(base)
                    else:
                        base_frames = list(base)

                    all_generated_frames = [Image.fromarray((f * 255).astype(np.uint8)) if isinstance(f, np.ndarray) and f.dtype != np.uint8 else Image.fromarray(f) if isinstance(f, np.ndarray) else f for f in base_frames]

                    current_video = all_generated_frames

                    # Run continuation segments
                    for seg_idx in range(generate_segments):
                        status_text.text(f"Generating segment {seg_idx+1}/{generate_segments}...")
                        # map width/height to standardized resolution string where appropriate
                        if width >= 700:
                            res_str = '720p'
                        elif width >= 480:
                            res_str = '480p'
                        else:
                            res_str = '256p'

                        out = pipeline.generate_vc(
                            video=current_video,
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            resolution=res_str,
                            num_frames=seg_num_frames,
                            num_cond_frames=num_cond_frames,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=generator,
                            use_kv_cache=True,
                            offload_kv_cache=False,
                            enhance_hf=True,
                        )

                        if isinstance(out, (list, tuple)):
                            out_v = out[0]
                        elif hasattr(out, "frames"):
                            out_v = out.frames[0]
                        else:
                            out_v = out

                        # Convert to PIL images
                        if isinstance(out_v, np.ndarray):
                            if out_v.ndim == 5:
                                new_video = [(out_v[0, i] * 255).astype(np.uint8) for i in range(out_v.shape[1])]
                            elif out_v.ndim == 4:
                                new_video = [(out_v[i] * 255).astype(np.uint8) for i in range(out_v.shape[0])]
                            else:
                                new_video = [ (frame * 255).astype(np.uint8) for frame in out_v ]
                        else:
                            new_video = out_v

                        new_video = [Image.fromarray(img) if isinstance(img, np.ndarray) else img for img in new_video]

                        # Append non-conditioning frames
                        all_generated_frames.extend(new_video[num_cond_frames:])

                        # Set current_video for next iteration
                        current_video = new_video

                    # After segments finished, set result to assembled frames
                    # Convert back to numpy array shape (N,H,W,C)
                    video_np = np.stack([np.array(f) for f in all_generated_frames], axis=0)
                    result = video_np

                else:
                    # Single-segment generation
                    result = pipeline.generate_t2v(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_frames=num_frames,
                        height=height,
                        width=width,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    )

            progress_bar.progress(90)
            status_text.text("Processing results...")

            # Handle direct return from pipeline
            # The pipeline returns numpy array with shape (B, N, H, W, C) when output_type="np"
            if isinstance(result, np.ndarray):
                print(
                    f"DEBUG: Pipeline result shape: {result.shape}, dtype: {result.dtype}"
                )
                video_frames = result
            elif hasattr(result, "frames"):
                video_frames = result.frames
            elif isinstance(result, (list, tuple)):
                video_frames = result
            else:
                # Assume it's already the video frames
                video_frames = result

            progress_bar.progress(100)
            status_text.text("✅ Generation completed!")

            # Display results
            with result_container:
                st.header("🎉 Generated Video")

                col1, col2 = st.columns([2, 1])

                with col1:
                    # Show first frame as preview
                    if video_frames is not None:
                        # Handle different frame formats for preview
                        first_frame = None

                        if isinstance(video_frames, np.ndarray):
                            if len(video_frames.shape) == 5:
                                # Shape (B, N, H, W, C)
                                first_frame = video_frames[0, 0]
                            elif len(video_frames.shape) == 4:
                                # Shape (N, H, W, C)
                                first_frame = video_frames[0]
                        elif (
                            isinstance(video_frames, (list, tuple))
                            and len(video_frames) > 0
                        ):
                            first_frame = video_frames[0]

                        if first_frame is not None:
                            # Convert to PIL Image for display if needed
                            if isinstance(first_frame, np.ndarray):
                                if first_frame.dtype != np.uint8:
                                    if first_frame.max() <= 1.0:
                                        first_frame = (first_frame * 255).astype(
                                            np.uint8
                                        )
                                    else:
                                        first_frame = np.clip(
                                            first_frame, 0, 255
                                        ).astype(np.uint8)
                                first_frame = Image.fromarray(first_frame)

                            st.image(
                                first_frame,
                                caption="First Frame Preview",
                                use_container_width=True,
                            )

                        # Save video
                        temp_file = tempfile.NamedTemporaryFile(
                            delete=False, suffix=".mp4"
                        )
                        if save_video(video_frames, temp_file.name):
                            # Provide download
                            with open(temp_file.name, "rb") as f:
                                st.download_button(
                                    label="💾 Download Video",
                                    data=f.read(),
                                    file_name=f"longcat_video_{num_frames}frames_{width}x{height}.mp4",
                                    mime="video/mp4",
                                    type="primary",
                                    use_container_width=True,
                                )

                            # Show video player
                            st.video(temp_file.name)
                        else:
                            st.error("Failed to save video")

                with col2:
                    st.subheader("📋 Generation Details")

                    # Get frame count safely
                    frame_count = 0
                    if video_frames is not None:
                        if isinstance(video_frames, np.ndarray):
                            if len(video_frames.shape) == 5:
                                # Shape (B, N, H, W, C)
                                frame_count = video_frames.shape[1]
                            elif len(video_frames.shape) == 4:
                                # Shape (N, H, W, C)
                                frame_count = video_frames.shape[0]
                        elif isinstance(video_frames, (list, tuple)):
                            frame_count = len(video_frames)

                    st.text(f"Frames: {frame_count}")
                    st.text(f"Resolution: {width}x{height}")
                    st.text(f"Inference Steps: {num_inference_steps}")
                    st.text(f"Guidance Scale: {guidance_scale}")
                    st.text(f"Seed: {seed}")

                    if frame_count > 0:
                        duration = frame_count / 8  # 8 FPS
                        st.text(f"Duration: ~{duration:.1f} seconds")

                        # Show all frames as a grid
                        st.subheader("🖼️ Frame Preview")
                        cols = st.columns(3)
                        frames_to_show = min(9, frame_count)

                        for i in range(frames_to_show):
                            # Get frame safely
                            frame = None
                            if isinstance(video_frames, np.ndarray):
                                if len(video_frames.shape) == 5:
                                    # Shape (B, N, H, W, C)
                                    frame = video_frames[0, i]
                                elif len(video_frames.shape) == 4:
                                    # Shape (N, H, W, C)
                                    frame = video_frames[i]
                            elif isinstance(video_frames, (list, tuple)):
                                frame = video_frames[i]

                            if frame is not None:
                                # Convert to PIL Image for display if needed
                                if isinstance(frame, np.ndarray):
                                    if frame.dtype != np.uint8:
                                        if frame.max() <= 1.0:
                                            frame = (frame * 255).astype(np.uint8)
                                        else:
                                            frame = np.clip(frame, 0, 255).astype(
                                                np.uint8
                                            )
                                    frame = Image.fromarray(frame)

                                with cols[i % 3]:
                                    st.image(
                                        frame,
                                        caption=f"Frame {i + 1}",
                                        use_container_width=True,
                                    )

        except Exception as e:
            st.error(f"❌ Generation failed: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()
