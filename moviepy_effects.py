from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, ImageClip
from moviepy.editor import CompositeVideoClip, concatenate_videoclips, CompositeAudioClip
import moviepy.video.fx.all as vfx
import moviepy.audio.fx.all as afx
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import json
import time

# ----- Helper Functions -----

def time_to_seconds(time_str):
    """Convert time string in format 'HH:MM:SS' to seconds."""
    t = time.strptime(time_str, "%H:%M:%S")
    return t.tm_hour * 3600 + t.tm_min * 60 + t.tm_sec

def seconds_to_time_str(seconds):
    """Convert seconds to time string in format 'HH:MM:SS'."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

# ----- Time-based Edits -----

def trim(clip, start_time, duration, **kwargs):
    """Trim the video to keep only the specified segment."""
    start_seconds = time_to_seconds(start_time)
    duration_seconds = time_to_seconds(duration)
    end_seconds = start_seconds + duration_seconds
    return clip.subclip(start_seconds, end_seconds)

def cut(clip, start_time, duration, **kwargs):
    """Cut out a segment from the video."""
    start_seconds = time_to_seconds(start_time)
    duration_seconds = time_to_seconds(duration)
    end_seconds = start_seconds + duration_seconds
    
    before = clip.subclip(0, start_seconds) if start_seconds > 0 else None
    after = clip.subclip(end_seconds, clip.duration) if end_seconds < clip.duration else None
    
    clips_to_concat = [c for c in [before, after] if c is not None]
    return concatenate_videoclips(clips_to_concat) if clips_to_concat else clip.subclip(0, 0)

def speed_up(clip, start_time, duration, speed_factor=1.5, **kwargs):
    """Speed up a segment of the video."""
    start_seconds = time_to_seconds(start_time)
    duration_seconds = time_to_seconds(duration)
    end_seconds = min(start_seconds + duration_seconds, clip.duration)
    
    # If start time is already past the clip duration, return original clip
    if start_seconds >= clip.duration:
        return clip
        
    # If start and end are the same (zero duration after adjustment), return original clip
    if start_seconds >= end_seconds:
        return clip
    
    if start_seconds <= 0 and end_seconds >= clip.duration:
        # Speed up the entire clip
        return clip.speedx(speed_factor)
    
    # Create subclips
    before = clip.subclip(0, start_seconds) if start_seconds > 0 else None
    middle = clip.subclip(start_seconds, end_seconds).speedx(speed_factor)
    after = clip.subclip(end_seconds, clip.duration) if end_seconds < clip.duration else None
    
    # Combine clips
    clips_to_concat = [c for c in [before, middle, after] if c is not None]
    return concatenate_videoclips(clips_to_concat)

def slow_down(clip, start_time, duration, speed_factor=0.5, **kwargs):
    """Slow down a segment of the video."""
    return speed_up(clip, start_time, duration, 1.0/float(speed_factor))

# ----- Visual Effects -----

def zoom_in(clip, start_time, duration, level=1.2, **kwargs):
    """Zoom in on a segment of the video."""
    start_seconds = time_to_seconds(start_time)
    duration_seconds = time_to_seconds(duration)
    end_seconds = start_seconds + duration_seconds
    
    def zoom_effect(get_frame, t):
        if t < start_seconds or t > end_seconds:
            return get_frame(t)
        
        # Calculate zoom factor (gradually zoom from 1.0 to level)
        progress = min(1.0, (t - start_seconds) / duration_seconds)
        zoom_factor = 1.0 + (float(level) - 1.0) * progress
        
        # Get frame and apply zoom
        frame = get_frame(t)
        h, w = frame.shape[:2]
        
        # Calculate new dimensions
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        
        # Crop center portion
        center_y, center_x = h//2, w//2
        y1 = max(0, center_y - h//2)
        y2 = min(new_h, center_y + h//2)
        x1 = max(0, center_x - w//2)
        x2 = min(new_w, center_x + w//2)
        
        # Resize and crop
        frame_pil = Image.fromarray(frame)
        zoomed = frame_pil.resize((new_w, new_h), Image.LANCZOS)
        zoomed_center = zoomed.crop((
            (new_w - w) // 2, 
            (new_h - h) // 2,
            (new_w + w) // 2,
            (new_h + h) // 2
        ))
        
        return np.array(zoomed_center)
    
    return clip.fl(zoom_effect)

def zoom_out(clip, start_time, duration, level=1.2, **kwargs):
    """Zoom out from a segment of the video."""
    start_seconds = time_to_seconds(start_time)
    duration_seconds = time_to_seconds(duration)
    end_seconds = start_seconds + duration_seconds
    
    def zoom_effect(get_frame, t):
        if t < start_seconds or t > end_seconds:
            return get_frame(t)
        
        # Calculate zoom factor (gradually zoom from level to 1.0)
        progress = min(1.0, (t - start_seconds) / duration_seconds)
        zoom_factor = float(level) - (float(level) - 1.0) * progress
        
        # Get frame and apply zoom
        frame = get_frame(t)
        h, w = frame.shape[:2]
        
        # Calculate new dimensions
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        
        # Resize and center in original frame
        frame_pil = Image.fromarray(frame)
        zoomed = frame_pil.resize((new_w, new_h), Image.LANCZOS)
        
        result = Image.new("RGB", (w, h))
        result.paste(zoomed, ((w - new_w) // 2, (h - new_h) // 2))
        
        return np.array(result)
    
    return clip.fl(zoom_effect)

def filter(clip, start_time, duration, type="boost", intensity=0.5, **kwargs):
    """Apply a filter to a segment of the video."""
    start_seconds = time_to_seconds(start_time)
    duration_seconds = time_to_seconds(duration)
    end_seconds = start_seconds + duration_seconds
    
    def filter_effect(get_frame, t):
        if t < start_seconds or t > end_seconds:
            return get_frame(t)
        
        frame = get_frame(t)
        img = Image.fromarray(frame)
        
        intensity_val = float(intensity)
        
        if type == "blur":
            img = img.filter(ImageFilter.GaussianBlur(radius=intensity_val * 5))
        elif type == "boost":
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.0 + intensity_val)
        elif type == "contrast":
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.0 + intensity_val)
        elif type == "darken":
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.0 - intensity_val * 0.5)
        elif type == "greyscale":
            img = img.convert("L").convert("RGB")
        elif type == "lighten":
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.0 + intensity_val * 0.5)
        elif type == "muted":
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(1.0 - intensity_val * 0.5)
        elif type == "negative":
            img = Image.fromarray(255 - np.array(img))
        
        return np.array(img)
    
    return clip.fl(filter_effect)

# ----- Transitions -----

def crossfade(clip1, clip2, duration, **kwargs):
    """Create a crossfade transition between two clips."""
    duration_seconds = time_to_seconds(duration)
    clip1 = clip1.crossfadeout(duration_seconds)
    clip2 = clip2.crossfadein(duration_seconds)
    return concatenate_videoclips([clip1, clip2])

def fade_in(clip, start_time, duration, **kwargs):
    """Apply a fade in effect to the beginning of the clip or at specific time."""
    duration_seconds = time_to_seconds(duration)
    start_seconds = time_to_seconds(start_time)
    
    if start_seconds <= 0:
        return clip.fadein(duration_seconds)
    
    # For fade in at specific time, we need to split and rejoin the clip
    before = clip.subclip(0, start_seconds)
    middle = clip.subclip(start_seconds, start_seconds + duration_seconds).fadein(duration_seconds)
    after = clip.subclip(start_seconds + duration_seconds) if start_seconds + duration_seconds < clip.duration else None
    
    clips_to_concat = [c for c in [before, middle, after] if c is not None]
    return concatenate_videoclips(clips_to_concat)

def fade_out(clip, start_time, duration, **kwargs):
    """Apply a fade out effect to the end of the clip or at specific time."""
    duration_seconds = time_to_seconds(duration)
    start_seconds = time_to_seconds(start_time)
    
    if start_seconds + duration_seconds >= clip.duration:
        return clip.fadeout(duration_seconds)
    
    before = clip.subclip(0, start_seconds) if start_seconds > 0 else None
    middle = clip.subclip(start_seconds, start_seconds + duration_seconds).fadeout(duration_seconds)
    after = clip.subclip(start_seconds + duration_seconds)
    
    clips_to_concat = [c for c in [before, middle, after] if c is not None]
    return concatenate_videoclips(clips_to_concat)

def wipe_transition(clip1, clip2, duration, direction="left", **kwargs):
    """Create a wipe transition between two clips."""
    # Implementation would depend on specific requirements
    # This is a placeholder for a wipe transition
    return crossfade(clip1, clip2, duration)

def slide_transition(clip1, clip2, duration, direction="left", **kwargs):
    """Create a slide transition between two clips."""
    # Implementation would depend on specific requirements
    # This is a placeholder for a slide transition
    return crossfade(clip1, clip2, duration)

# ----- Overlays -----

def overlay_text(clip, start_time, duration, text="Sample Text", position="center", 
                fontsize=48, color="white", font="Arial", **kwargs):
    """Add text overlay to the video."""
    start_seconds = time_to_seconds(start_time)
    duration_seconds = time_to_seconds(duration)
    
    txt_clip = TextClip(text, fontsize=fontsize, color=color, font=font)
    txt_clip = txt_clip.set_position(position).set_start(start_seconds).set_duration(duration_seconds)
    
    return CompositeVideoClip([clip, txt_clip])

def caption(clip, start_time, duration, text="Caption", **kwargs):
    """Add caption to the bottom of the video."""
    return overlay_text(clip, start_time, duration, text=text, position=("center", "bottom"), fontsize=32)

def overlay_image(clip, start_time, duration, image_path, position="center", opacity=1.0, **kwargs):
    """Add image overlay to the video."""
    start_seconds = time_to_seconds(start_time)
    duration_seconds = time_to_seconds(duration)
    
    img_clip = ImageClip(image_path).set_opacity(opacity)
    img_clip = img_clip.set_position(position).set_start(start_seconds).set_duration(duration_seconds)
    
    return CompositeVideoClip([clip, img_clip])

def watermark(clip, image_path, position="bottom-right", opacity=0.5, **kwargs):
    """Add watermark to the entire video."""
    start_time = "00:00:00"
    duration = seconds_to_time_str(clip.duration)
    return overlay_image(clip, start_time, duration, image_path, position=position, opacity=opacity)

# ----- Audio Effects -----

def audio_fade_in(clip, start_time, duration, **kwargs):
    """Apply audio fade in effect to the clip."""
    duration_seconds = time_to_seconds(duration)
    start_seconds = time_to_seconds(start_time)
    
    if start_seconds <= 0:
        return clip.audio_fadein(duration_seconds)
    
    # For fade in at specific time, apply audio transformation
    def adjust_volume(t):
        if t < start_seconds:
            return 1.0
        elif t < start_seconds + duration_seconds:
            return (t - start_seconds) / duration_seconds
        else:
            return 1.0
    
    clip = clip.copy()
    clip.audio = clip.audio.volumex(adjust_volume)
    return clip

def audio_fade_out(clip, start_time, duration, **kwargs):
    """Apply audio fade out effect to the clip."""
    duration_seconds = time_to_seconds(duration)
    start_seconds = time_to_seconds(start_time)
    
    if start_seconds + duration_seconds >= clip.duration:
        return clip.audio_fadeout(duration_seconds)
    
    # For fade out at specific time, apply audio transformation
    def adjust_volume(t):
        if t < start_seconds:
            return 1.0
        elif t < start_seconds + duration_seconds:
            return 1.0 - (t - start_seconds) / duration_seconds
        else:
            return 0.0
    
    clip = clip.copy()
    clip.audio = clip.audio.volumex(adjust_volume)
    return clip

def audio_ducking(clip, start_time, duration, level=0.3, **kwargs):
    """Reduce audio volume during a segment of the video."""
    start_seconds = time_to_seconds(start_time)
    duration_seconds = time_to_seconds(duration)
    end_seconds = start_seconds + duration_seconds
    
    def adjust_volume(t):
        if start_seconds <= t <= end_seconds:
            return float(level)
        return 1.0
    
    clip = clip.copy()
    clip.audio = clip.audio.volumex(adjust_volume)
    return clip

def add_background_music(clip, audio_path, volume=0.5, fade_in_duration=None, fade_out_duration=None, **kwargs):
    """Add background music to the video."""
    audio_clip = AudioFileClip(audio_path).volumex(float(volume))
    
    # Loop audio if needed to match video duration
    if audio_clip.duration < clip.duration:
        audio_clip = afx.audio_loop(audio_clip, duration=clip.duration)
    else:
        audio_clip = audio_clip.set_duration(clip.duration)
    
    # Apply fades if specified
    if fade_in_duration:
        audio_clip = audio_clip.audio_fadein(time_to_seconds(fade_in_duration))
    if fade_out_duration:
        audio_clip = audio_clip.audio_fadeout(time_to_seconds(fade_out_duration))
    
    # Mix with original audio
    final_audio = CompositeAudioClip([clip.audio, audio_clip])
    return clip.set_audio(final_audio)

def sound_effect(clip, start_time, audio_path, volume=1.0, **kwargs):
    """Add sound effect at a specific time in the video."""
    start_seconds = time_to_seconds(start_time)
    
    effect_clip = AudioFileClip(audio_path).volumex(float(volume))
    effect_clip = effect_clip.set_start(start_seconds)
    
    # Mix with original audio
    final_audio = CompositeAudioClip([clip.audio, effect_clip])
    return clip.set_audio(final_audio)

# ----- Main Processing Function -----

def trim_and_join(original_clip, trim_segments):
    """
    Extract multiple segments from the original clip and join them together.
    
    Args:
        original_clip: The complete original video clip
        trim_segments: List of dicts with start_time and duration for each segment to keep
    
    Returns:
        Concatenated video of all selected segments
    """
    segments = []
    
    # First filter out any segments that are completely beyond the clip duration
    valid_segments = [
        segment for segment in trim_segments 
        if time_to_seconds(segment["start_time"]) < original_clip.duration
    ]
    
    if not valid_segments:
        print("No valid segments found within clip duration!")
        return original_clip
        
    print(f"Processing {len(valid_segments)} valid trim segments out of {len(trim_segments)} total")
    
    for segment in valid_segments:
        start_time = segment["start_time"]
        duration = segment["duration"]
        
        start_seconds = time_to_seconds(start_time)
        duration_seconds = time_to_seconds(duration)
        end_seconds = min(start_seconds + duration_seconds, original_clip.duration)
        
        # Skip segments with zero duration after adjustment
        if end_seconds <= start_seconds:
            print(f"Warning: Skipping segment at {start_time} - zero duration after adjustment")
            continue
        
        # Extract the segment
        print(f"Extracting segment from {start_time} to {seconds_to_time_str(end_seconds)}")
        segment_clip = original_clip.subclip(start_seconds, end_seconds)
        segments.append(segment_clip)
    
    if not segments:
        print("No valid segments to join!")
        return original_clip
    
    # Join all segments together
    return concatenate_videoclips(segments)

def apply_edits_from_json(video_path, json_path, output_path):
    """Apply edits from JSON file to a video."""
    # Load the video
    try:
        clip = VideoFileClip(video_path)
        original_clip = clip.copy()  # Keep a copy of the original
    except Exception as e:
        print(f"Error loading video file '{video_path}': {str(e)}")
        return False
    
    # Load the edit instructions
    try:
        with open(json_path, 'r') as f:
            edits = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file '{json_path}': {str(e)}")
        clip.close()
        return False
    
    # Check if this is primarily a trim operation (extracting segments)
    trim_operations = [edit for edit in edits if edit["edit_type"] == "trim"]
    other_operations = [edit for edit in edits if edit["edit_type"] != "trim"]
    
    if len(trim_operations) > 1:
        print(f"Detected {len(trim_operations)} trim operations - processing as segment extraction")
        clip = trim_and_join(original_clip, trim_operations)
        
        # If there are other operations to apply after segment extraction
        if other_operations:
            print(f"Applying {len(other_operations)} additional operations to the extracted segments")
            # Continue with the remaining edits on the extracted segments
            edits = other_operations
        else:
            # No further edits needed
            print("No additional operations to apply")
            # Write the final video and return
            try:
                clip.write_videofile(output_path)
                print(f"Successfully created {output_path}")
                clip.close()
                return True
            except Exception as e:
                print(f"Error writing output video to '{output_path}': {str(e)}")
                clip.close()
                return False
    
    # Function mapping for regular editing operations
    edit_functions = {
        'trim': trim,
        'cut': cut,
        'zoom_in': zoom_in,
        'zoom_out': zoom_out,
        'crossfade': crossfade,
        'fade_in': fade_in,
        'fade_out': fade_out,
        'wipe': wipe_transition,
        'slide_transition': slide_transition,
        'overlay_text': overlay_text,
        'caption': caption,
        'overlay_image': overlay_image,
        'watermark': watermark,
        'filter': filter,
        'speed_up': speed_up,
        'slow_down': slow_down,
        'audio_fade_in': audio_fade_in,
        'audio_fade_out': audio_fade_out,
        'audio_ducking': audio_ducking,
        'add_background_music': add_background_music,
        'sound_effect': sound_effect
    }
    
    # Apply each edit
    for edit in edits:
        edit_type = edit["edit_type"]
        start_time = edit["start_time"]
        duration = edit["duration"]
        
        # Convert parameters list to kwargs dictionary
        kwargs = {}
        parameters = edit["parameters"]
        for i in range(0, len(parameters), 2):
            if i+1 < len(parameters):
                kwargs[parameters[i]] = parameters[i+1]
        
        # Convert times to seconds for validation
        start_seconds = time_to_seconds(start_time)
        duration_seconds = time_to_seconds(duration)
        end_seconds = start_seconds + duration_seconds
        
        # Check if the edit is within the clip's duration
        if start_seconds >= clip.duration:
            print(f"Warning: Skipping {edit_type} - start time {start_time} ({start_seconds}s) is beyond clip duration ({clip.duration:.2f}s)")
            continue
            
        # Adjust end time if it exceeds the clip's duration
        if end_seconds > clip.duration:
            print(f"Warning: Adjusting duration for {edit_type} - end time exceeds clip duration")
            duration_seconds = max(0, clip.duration - start_seconds)
            duration = seconds_to_time_str(duration_seconds)
        
        # Apply the edit if function exists
        if edit_type in edit_functions:
            print(f"Applying {edit_type} from {start_time} for {duration} with params: {kwargs}")
            clip = edit_functions[edit_type](clip, start_time, duration, **kwargs)
        else:
            print(f"Warning: Edit type '{edit_type}' not implemented")
    
    # Write the final video
    try:
        clip.write_videofile(output_path)
        print(f"Successfully created {output_path}")
        result = True
    except Exception as e:
        print(f"Error writing output video to '{output_path}': {str(e)}")
        result = False
    
    clip.close()
    return result

if __name__ == "__main__":
    import sys
    import os
    
    # Default values
    video_path = "video.mp4"
    json_path = "shotstack_hp.json"
    output_path = "output.mp4"
    
    # Check for command-line arguments
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    if len(sys.argv) > 2:
        json_path = sys.argv[2]
    if len(sys.argv) > 3:
        output_path = sys.argv[3]
    
    # Check if input files exist
    if not os.path.exists(video_path):
        print(f"Error: Input video file '{video_path}' not found")
        sys.exit(1)

    success = apply_edits_from_json(video_path, json_path, output_path)