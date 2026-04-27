import os
import argparse
import json
import sys
import cv2 # Added for video processing
import numpy as np # Added for array handling
from tqdm import tqdm
from backend import QwenEngine, SAM3Engine
from PIL import Image, ImageOps
from file_utils import find_media_files, IMAGE_EXTS, VIDEO_EXTS

SETTINGS_FILE = "settings.json"

def parse_quant_string(q_str):
    """Maps the verbose GUI quantization string to the CLI short code."""
    if not isinstance(q_str, str): return "None"
    if "FP16" in q_str: return "FP16"
    if "Int8" in q_str: return "Int8"
    if "NF4" in q_str: return "NF4"
    return "None"

RES_MAP = {
    0: 336,
    1: 512,
    2: 768,
    3: 1024,
    4: 1280
}

def load_defaults_from_settings():
    """Reads the GUI settings file to populate defaults."""
    if not os.path.exists(SETTINGS_FILE):
        return {}
    
    try:
        with open(SETTINGS_FILE, "r") as f:
            data = json.load(f)
        
        gen_defaults = {
            "model_index": data.get("generate_tab", {}).get("model_index", 0),
            "quant": parse_quant_string(data.get("generate_tab", {}).get("quantization", "None")),
            "res": RES_MAP.get(data.get("generate_tab", {}).get("resolution_idx", 1), 512),
            "batch_size": data.get("generate_tab", {}).get("batch_size", 16),
            "frame_count": data.get("generate_tab", {}).get("frames", 8),
            "max_tokens": data.get("generate_tab", {}).get("tokens", 384),
            "trigger": data.get("generate_tab", {}).get("trigger", ""),
            "prompt": data.get("generate_tab", {}).get("prompt_text", ""),
            "prompt_suffix": data.get("generate_tab", {}).get("suffix", ""),
            "skip_existing": data.get("generate_tab", {}).get("skip_existing", False),
            "vision_tokens": data.get("generate_tab", {}).get("vision_tokens", None)
        }

        if "generate_tab" not in data:
            gen_defaults["folder"] = data.get("folder", "")
        else:
            gen_defaults["folder"] = data.get("folder", "")

        # Mask Defaults
        mask_data = data.get("mask_tab", {})
        mask_defaults = {
            "mask_prompt": mask_data.get("prompt", ""),
            "mask_res": mask_data.get("max_res", 1024),
            "mask_expand": mask_data.get("expand_percent", 3.0),
            "mask_skip": mask_data.get("skip_existing", True),
            "mask_crop": mask_data.get("crop_to_mask", False)
        }
        
        # Video Defaults (New)
        video_data = data.get("video_tab", {})
        video_defaults = {
            "video_step": video_data.get("step", 30),
            "video_start": video_data.get("start_frame", 0),
            "video_end": video_data.get("end_frame", -1),
            "video_res": video_data.get("res", 1024),
            "video_conf": video_data.get("conf", 0.25),
            "video_expand": video_data.get("expand", 2.0),
            "video_crop": video_data.get("crop", False)
        }
        
        # Merge dicts
        return {**gen_defaults, **mask_defaults, **video_defaults}

    except Exception as e:
        print(f"⚠️ Warning: Failed to read {SETTINGS_FILE}: {e}")
        return {}

def get_model_path_from_index(index):
    """Attempts to find the model directory based on index."""
    base_path = os.path.join(os.getcwd(), "models")
    if not os.path.exists(base_path):
        return None
    
    models = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    if 0 <= index < len(models):
        return os.path.join(base_path, models[index])
    return None

def find_images_in_folder(folder, recursive=False):
    """Helper to just get image files."""
    return find_media_files(folder, exts=IMAGE_EXTS, recursive=recursive)

def find_videos_in_folder(folder, recursive=False):
    """Helper to find video files."""
    # If the user passed a file path instead of a folder, return just that file
    if os.path.isfile(folder):
        return [folder]
    return find_media_files(folder, exts=VIDEO_EXTS, recursive=recursive)

def main():
    defaults = load_defaults_from_settings()
    
    parser = argparse.ArgumentParser(description="VisionCaptioner CLI")
    
    # Global Config
    parser.add_argument("--folder", type=str, default=defaults.get("folder"), help="Path to folder containing images/videos (or a single video file).")
    parser.add_argument("--output", type=str, default=None, help="Optional output folder for video extraction. Defaults to input folder.")
    parser.add_argument("--mode", type=str, default="caption", choices=["caption", "mask", "video"], help="Operation mode: 'caption' (Qwen), 'mask' (SAM3), or 'video' (Extract & Mask).")
    parser.add_argument("--skip-existing", action="store_true", help="Skip files that already have results.")
    parser.add_argument("--recursive", "-r", action="store_true", help="Recursively scan subdirectories for images/videos.")

    # Captioning Config (Qwen)
    grp_cap = parser.add_argument_group("Captioning Arguments")
    grp_cap.add_argument("--model", type=str, help="Path to Qwen model.")
    grp_cap.add_argument("--quant", type=str, default=defaults.get("quant", "None"), choices=["None", "FP16", "Int8", "NF4"], help="Quantization level.")
    grp_cap.add_argument("--res", type=int, default=defaults.get("res", 512), help="Max resolution for captioning.")
    grp_cap.add_argument("--batch-size", type=int, default=defaults.get("batch_size", 4), help="Batch size.")
    grp_cap.add_argument("--frame-count", type=int, default=defaults.get("frame_count", 8), help="Video frame count (for captioning).")
    grp_cap.add_argument("--max-tokens", type=int, default=defaults.get("max_tokens", 384), help="Max output tokens.")
    grp_cap.add_argument("--prompt", type=str, default=defaults.get("prompt", "Describe this image."), help="System prompt.")
    grp_cap.add_argument("--suffix", type=str, default=defaults.get("prompt_suffix", ""), help="Suffix to append to prompt.")
    grp_cap.add_argument("--trigger", type=str, default=defaults.get("trigger", ""), help="Trigger word to prepend to caption.")
    grp_cap.add_argument("--vision-tokens", type=int, default=defaults.get("vision_tokens"), choices=[70, 140, 280, 560, 1120], help="Gemma 4 vision token budget per image (ignored for Qwen).")
    
    # Masking Config (SAM3)
    grp_mask = parser.add_argument_group("Masking Arguments")
    grp_mask.add_argument("--mask-prompt", type=str, default=defaults.get("mask_prompt", ""), help="Text prompt for object to mask (Required for mask/video mode).")
    grp_mask.add_argument("--mask-res", type=int, default=defaults.get("mask_res", 1024), help="Processing resolution for SAM3.")
    grp_mask.add_argument("--mask-expand", type=float, default=defaults.get("mask_expand", 3.0), help="Mask expansion percentage (0-50).")
    grp_mask.add_argument("--crop-to-mask", action="store_true", help="Crops image and mask to the mask's bounding box.")

    # Video Extraction Config
    grp_vid = parser.add_argument_group("Video Extraction Arguments")
    grp_vid.add_argument("--video-step", type=int, default=defaults.get("video_step", 30), help="Extract every Nth frame.")
    grp_vid.add_argument("--video-start", type=int, default=defaults.get("video_start", 0), help="Start frame index.")
    grp_vid.add_argument("--video-end", type=int, default=defaults.get("video_end", -1), help="End frame index (-1 for end of video).")
    grp_vid.add_argument("--video-conf", type=float, default=defaults.get("video_conf", 0.25), help="Confidence threshold for SAM3.")

    args = parser.parse_args()

    # --- VALIDATION ---
    if not args.folder or not os.path.exists(args.folder):
        print("❌ Error: Invalid or missing --folder argument.")
        return
    
    # Apply defaults for skip_existing/crop if flag not strictly passed but setting existed
    if args.mode == "caption":
        if defaults.get("skip_existing") is True and not args.skip_existing:
             args.skip_existing = True
    elif args.mode == "mask":
        if defaults.get("mask_skip") is True and not args.skip_existing:
            args.skip_existing = True
        if defaults.get("mask_crop") is True and not args.crop_to_mask:
            args.crop_to_mask = True
    elif args.mode == "video":
        if defaults.get("video_crop") is True and not args.crop_to_mask:
            args.crop_to_mask = True


    # ==============================================================================
    # MODE: CAPTION
    # ==============================================================================
    if args.mode == "caption":
        # ... (Same as existing code)
        model_path = None
        if args.model:
            if os.path.exists(args.model):
                model_path = args.model
            else:
                rel_path = os.path.join(os.getcwd(), "models", args.model)
                if os.path.exists(rel_path):
                    model_path = rel_path
                else:
                    print(f"❌ Error: Model path '{args.model}' not found.")
                    return
        else:
            idx = defaults.get("model_index", 0)
            model_path = get_model_path_from_index(idx)
            if not model_path:
                print("❌ Error: No model specified and could not infer from settings.")
                return

        final_prompt = args.prompt
        if args.suffix.strip():
            final_prompt += "\n" + args.suffix

        print(f"\n🚀 --- VisionCaptioner CLI (Caption Mode) ---")
        print(f"📁 Folder:   {args.folder}")
        print(f"🧠 Model:    {os.path.basename(model_path)}")
        print(f"⚙️  Settings: Res={args.res}, Quant={args.quant}, Batch={args.batch_size}")
        print(f"📝 Prompt:   {final_prompt[:50]}...")
        if args.skip_existing:
            print("⏩ Skipping existing .txt files.")
        print("-" * 40)

        engine = QwenEngine()
        print("🔍 Scanning folder...")
        all_pairs = engine.find_files(args.folder, skip_existing=args.skip_existing, recursive=args.recursive)
        
        if not all_pairs:
            print("❌ No files found (or all skipped).")
            return
        
        print(f"✅ Found {len(all_pairs)} files to process.")

        video_exts_set = tuple(e.lstrip("*").lower() for e in VIDEO_EXTS)
        has_video = any(f.lower().endswith(video_exts_set) for f, _ in all_pairs)
        media_mode = "video" if has_video else "image"

        success, msg = engine.load_model(model_path, quantization_type=args.quant, max_resolution=args.res, vision_token_budget=args.vision_tokens, media_mode=media_mode)
        if not success:
            print(f"❌ Model Load Failed: {msg}")
            return

        try:
            total = len(all_pairs)
            batch_size = args.batch_size
            
            with tqdm(total=total, unit="img") as pbar:
                for i in range(0, total, batch_size):
                    batch_pairs = all_pairs[i : i + batch_size]
                    final_files = []
                    final_masks = []
                    
                    for f, m in batch_pairs:
                        final_files.append(f)
                        final_masks.append(m)

                    if not final_files:
                        pbar.update(len(batch_pairs))
                        continue

                    captions = engine.generate_batch(
                        final_files,
                        prompt_text=final_prompt,
                        trigger_word=args.trigger,
                        frame_count=args.frame_count,
                        mask_paths=final_masks,
                        max_tokens=args.max_tokens,
                        log_callback=None
                    )

                    for idx, f_path in enumerate(final_files):
                        cap = captions[idx]
                        if "Error:" in cap or "[Video Load Error]" in cap:
                            continue
                        txt_path = os.path.splitext(f_path)[0] + ".txt"
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(cap)
                    
                    pbar.update(len(batch_pairs))

        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user.")
        except Exception as e:
            print(f"\n❌ Unexpected Error: {e}")
        finally:
            print("\n🧹 Unloading model...")
            engine.unload_model()
            print("👋 Done.")

    # ==============================================================================
    # MODE: MASK
    # ==============================================================================
    elif args.mode == "mask":
        if not args.mask_prompt:
            print("❌ Error: --mask-prompt is required for mask mode.")
            return

        sam_engine = SAM3Engine()
        if not sam_engine.is_available():
            print("❌ Error: SAM3 library not installed or not found.")
            return

        model_path = os.path.join(os.getcwd(), "models", "sam3")
        if not os.path.exists(model_path):
             print(f"❌ Error: SAM3 model folder not found at {model_path}")
             return

        print(f"\n🚀 --- VisionCaptioner CLI (Mask Mode) ---")
        print(f"📁 Folder:   {args.folder}")
        print(f"🎯 Prompt:   '{args.mask_prompt}'")
        print(f"⚙️  Settings: Res={args.mask_res}, Expand={args.mask_expand}%, Crop={args.crop_to_mask}")
        if args.skip_existing:
            print("⏩ Skipping existing *-masklabel.png files.")
        print("-" * 40)

        print("⏳ Loading SAM3 Model...")
        success, msg = sam_engine.load_model(model_path)
        if not success:
            print(f"❌ Model Load Failed: {msg}")
            return
        
        all_files = find_images_in_folder(args.folder, recursive=args.recursive)
        
        files_to_process = []
        skipped_count = 0
        
        for f in all_files:
            base_name = os.path.splitext(f)[0]
            mask_path = f"{base_name}-masklabel.png"
            if args.skip_existing and os.path.exists(mask_path):
                skipped_count += 1
            else:
                files_to_process.append(f)

        if skipped_count > 0:
            print(f"⏩ Skipped {skipped_count} existing masks.")
        
        if not files_to_process:
            print("✅ No files left to process.")
            sam_engine.unload()
            return

        print(f"✅ Processing {len(files_to_process)} images.")

        try:
            expand_ratio = args.mask_expand / 100.0
            
            with tqdm(total=len(files_to_process), unit="img") as pbar:
                for f_path in files_to_process:
                    base_name = os.path.splitext(f_path)[0]
                    save_path = f"{base_name}-masklabel.png"
                    
                    if args.skip_existing and os.path.exists(save_path):
                        pbar.update(1)
                        continue

                    mask_img, msg = sam_engine.generate_mask(
                        f_path,
                        prompt=args.mask_prompt,
                        max_dimension=args.mask_res,
                        conf_threshold=0.25, 
                        expand_ratio=expand_ratio
                    )

                    if mask_img:
                        try:
                            if args.crop_to_mask:
                                bbox = mask_img.getbbox()
                                if bbox:
                                    original_pil_img = Image.open(f_path)
                                    original_pil_img = ImageOps.exif_transpose(original_pil_img)
                                    original_pil_img = original_pil_img.convert("RGB")

                                    cropped_img = original_pil_img.crop(bbox)
                                    cropped_mask = mask_img.crop(bbox)

                                    uncropped_dir = os.path.join(args.folder, "uncropped")
                                    os.makedirs(uncropped_dir, exist_ok=True)
                                    backup_path = os.path.join(uncropped_dir, os.path.basename(f_path))
                                    original_pil_img.save(backup_path)

                                    cropped_img.save(f_path)
                                    cropped_mask.save(save_path)
                                else:
                                    mask_img.save(save_path)
                            else:
                                mask_img.save(save_path)
                        except Exception as e:
                            print(f"\n❌ Error saving/cropping {os.path.basename(save_path)}: {e}")
                    
                    pbar.update(1)

        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user.")
        except Exception as e:
            print(f"\n❌ Unexpected Error: {e}")
        finally:
            print("\n🧹 Unloading model...")
            sam_engine.unload()
            print("👋 Done.")

    # ==============================================================================
    # MODE: VIDEO (EXTRACT)
    # ==============================================================================
    elif args.mode == "video":
        if not args.mask_prompt:
            print("❌ Error: --mask-prompt is required for video extraction (used to detect/crop objects).")
            return

        sam_engine = SAM3Engine()
        if not sam_engine.is_available():
            print("❌ Error: SAM3 library not installed or not found.")
            return

        model_path = os.path.join(os.getcwd(), "models", "sam3")
        if not os.path.exists(model_path):
             print(f"❌ Error: SAM3 model folder not found at {model_path}")
             return

        # Determine output folder
        output_folder = args.output if args.output else args.folder
        if os.path.isfile(args.folder) and args.output is None:
            output_folder = os.path.dirname(args.folder)
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        print(f"\n🚀 --- VisionCaptioner CLI (Video Extract Mode) ---")
        print(f"🎬 Input:    {args.folder}")
        print(f"💾 Output:   {output_folder}")
        print(f"🎯 Prompt:   '{args.mask_prompt}'")
        print(f"⚙️  Settings: Start={args.video_start}, End={args.video_end}, Step={args.video_step}")
        print(f"📐 Process:  Crop={args.crop_to_mask}, Expand={args.mask_expand}%, Conf={args.video_conf}")
        print("-" * 40)

        # 1. Load SAM3
        print("⏳ Loading SAM3 Model...")
        success, msg = sam_engine.load_model(model_path)
        if not success:
            print(f"❌ Model Load Failed: {msg}")
            return

        # 2. Find Videos
        videos = find_videos_in_folder(args.folder, recursive=args.recursive)
        if not videos:
            print("❌ No video files found.")
            sam_engine.unload()
            return

        print(f"✅ Found {len(videos)} videos.")
        
        expand_ratio = args.mask_expand / 100.0

        try:
            total_saved = 0
            
            for v_idx, video_path in enumerate(videos):
                print(f"\n🎬 Processing Video [{v_idx+1}/{len(videos)}]: {os.path.basename(video_path)}")
                
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    print(f"⚠️ Could not open {video_path}")
                    continue

                total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Determine loop range
                start_f = args.video_start
                end_f = args.video_end
                if end_f == -1 or end_f >= total_frames_in_video:
                    end_f = total_frames_in_video - 1
                
                # Setup progress bar
                frames_to_scan = range(start_f, end_f + 1, args.video_step)
                base_filename = os.path.splitext(os.path.basename(video_path))[0]

                with tqdm(total=len(frames_to_scan), unit="fr") as pbar:
                    for frame_idx in frames_to_scan:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        if not ret: break

                        # Convert BGR (OpenCV) to RGB (PIL)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(frame_rgb)
                        
                        # Generate Mask
                        mask_img, msg = sam_engine.generate_mask(
                            pil_img, 
                            prompt=args.mask_prompt,
                            max_dimension=args.mask_res,
                            conf_threshold=args.video_conf,
                            expand_ratio=expand_ratio
                        )

                        if mask_img:
                            save_img = pil_img
                            save_mask = mask_img
                            suffix = ""

                            # Crop logic
                            if args.crop_to_mask:
                                bbox = mask_img.getbbox()
                                if bbox:
                                    save_img = pil_img.crop(bbox)
                                    save_mask = mask_img.crop(bbox)
                                else:
                                    suffix = "_empty"
                            
                            # Construct filenames
                            frame_name = f"{base_filename}_frame_{frame_idx:06d}{suffix}.jpg"
                            mask_name = f"{base_filename}_frame_{frame_idx:06d}{suffix}-masklabel.png"
                            
                            out_path_img = os.path.join(output_folder, frame_name)
                            out_path_mask = os.path.join(output_folder, mask_name)
                            
                            # Save
                            try:
                                save_img.save(out_path_img, quality=95)
                                save_mask.save(out_path_mask)
                                total_saved += 1
                            except Exception as e:
                                print(f"Error saving frame {frame_idx}: {e}")

                        pbar.update(1)

                cap.release()
            
            print(f"\n✅ Extraction Complete. Saved {total_saved} pairs.")

        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user.")
        except Exception as e:
            print(f"\n❌ Unexpected Error: {e}")
        finally:
            print("🧹 Unloading model...")
            sam_engine.unload()
            print("👋 Done.")

if __name__ == "__main__":
    main()