# Captions tab

The Captions tab is used to generate captions for images and videos.

![Screenshot Captions Tab](screenshots/captions.jpg)

## General Use ##
* First select a folder with images or videos to process at the top of the window.

* To get started make sure to select a model from the dropdown menu and hit "Load Model".
Models can be downloaded using the built-in downloads manager on the Captions tab (📥💾 button), more info in the [readme_models.md](readme_models.md) file.

* Select a "System Prompt" from the dropdown menu or use the "Custom" option to enter your own system prompt.
* Set any other options as desired (they all have tooltips to explain what they do).
* Hit "TEST FIRST IMAGE" or "TEST RANDOM IMAGE" to test on a single image.
* Hit "START PROCESSING" to process all images in the selected folder.

In your output folder you will find a .txt file for each image/video with the generated caption.

## Optimizing Speed ##
* Although a single image can be a bit slow (ranging from a few seconds to tens of seconds)
* Increasing the "Batch Size" will allow multiple images to be processed at once, experiment by raising this value to as high as your VRAM allows
* Lowering precision and/or selecting a different attention implementation and/or torch compile may increase speed (only supported for non-GGUF models)
* Some GGUF models may be faster as well, but require manual downloading and installation of the llama-cpp-python package, see above for instructions. This is for more advanced users.

## Customizing System Prompt ##
* System prompt files can be found as text files in the "prompts" folder
* The app checks these at startup
* You can select a predefined prompt from the list
* Or you can use the "Custom" setting and type your system prompt in the app
* Or create/edit the text files in the "prompts" folder
