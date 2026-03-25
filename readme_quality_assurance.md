# Quality Assurance tab

The Quality Assurance tab analyzes all images in your folder for quality issues and lets you take action on problematic images.
This helps curate training datasets by identifying blurry, low-resolution, or otherwise problematic images before captioning.

## Analysis Criteria

The tab scores each image on the following criteria (each can be enabled/disabled):

* **Blur Detection** - Uses Laplacian variance to detect blurry or out-of-focus images. Configurable low/high thresholds determine what counts as "definitely blurry" vs "definitely sharp".
* **Low Resolution** - Flags images where the shortest dimension falls below a configurable minimum (default 512px).
* **Missing Mask** - Checks if a corresponding mask file exists. Automatically disabled if no images in the folder have masks.
* **Face Detection** - Detects faces in images. Configurable mode: flag if no face is detected (for portrait datasets) or flag if a face is detected (for landscape/object datasets).
* **Eyes Closed** - Detects closed eyes on detected faces using MediaPipe Face Mesh blendshapes. Falls back to OpenCV Haar cascades if MediaPipe is not installed.

## Scoring

Each criterion produces a score from 0.0 (worst) to 1.0 (best).
These are combined into an overall score using a weighted average.
Weights for each criterion can be adjusted using the sliders (0.0 to 2.0 range).

## Table View

The image list is displayed as a sortable table with columns for each score.
Click any column header to sort by that criterion.
Default sort is by Overall score descending (best at top).

## Navigation

* Click an image in the table to view it
* Use the slider below the image to scrub through the list
* Use **Left/Right arrow keys** to step through images
* Use the **mouse scroll wheel** on the image to navigate

## Per-Image Actions

* **Append to Caption [A]** - Appends the configured text to the image's caption file
* **Move to Unused [D]** - Moves the image (along with its caption and mask files) to the `unused/` subfolder

## Batch Actions

Apply actions to all images that fall below a score threshold:

1. Select which **Criterion** to threshold on (Overall, Blur, Resolution, Mask, Face, or Eyes)
2. Set the **Threshold** slider (0.0 to 1.0)
3. The count label shows how many images fall below the threshold
4. Click **Append to All Below** to add quality tags to their captions
5. Click **Move All Below to Unused** to move them (with their captions and masks) to the `unused/` subfolder

## Append Text Presets

The text to append can be typed freely or selected from presets:
* `, low quality, worst quality, jpeg artifacts, blurry`
* `, blurry, out of focus`
* `, low resolution, pixelated`
* `, eyes closed`

## Caching

Analysis results are cached to a `_qa_cache.json` file in the image folder.
When you reopen the same folder, cached results are loaded automatically if the files haven't changed.
Click **Analyze All** to re-scan and overwrite the cache.

## Models

On first analysis with face/eyes detection enabled, small model files are automatically downloaded:
* **MediaPipe Face Landmarker** (~3.7 MB) - for face and eye-blink detection
* **YuNet** (~336 KB) - OpenCV face detection fallback (used only if MediaPipe is not available)

These are saved to the `models/_/` directory.
