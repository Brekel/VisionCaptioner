import os
import inspect


class BaseFamily:
	name = "base"
	# Whether the processor expects a flat image list (True) or nested per-sample lists (False).
	flatten_vision_inputs = True

	def processor_kwargs(self, max_resolution=768, vision_token_budget=None):
		return {}

	def load_kwargs(self):
		return {}

	def build_content_block(self, is_video, pil_image, pil_frames, prompt_text):
		content = []
		if is_video:
			content.append({"type": "video", "video": pil_frames})
		else:
			content.append({"type": "image", "image": pil_image})
		content.append({"type": "text", "text": prompt_text})
		return content

	def apply_template(self, processor, messages):
		return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

	def extract_vision_inputs(self, messages):
		images = []
		videos = []
		for msg in messages:
			content = msg.get("content")
			if not isinstance(content, list):
				continue
			for item in content:
				if not isinstance(item, dict):
					continue
				t = item.get("type")
				if t == "image":
					val = item.get("image") or item.get("url")
					if val is not None:
						images.append(val)
				elif t == "video":
					val = item.get("video") or item.get("url")
					if val is not None:
						videos.append(val)
		return (images or None), (videos or None)

	def clean_output(self, processor, decoded_text):
		return decoded_text

	def decode_skip_special_tokens(self):
		return True


class QwenFamily(BaseFamily):
	name = "qwen"

	def processor_kwargs(self, max_resolution=768, vision_token_budget=None):
		total_pixels = max_resolution * max_resolution
		min_pixels = 256 * 28 * 28
		return {"min_pixels": min_pixels, "max_pixels": total_pixels}

	def extract_vision_inputs(self, messages):
		# Use qwen-vl-utils on the Qwen path so behavior stays identical to before.
		from qwen_vl_utils import process_vision_info
		return process_vision_info(messages)


class Gemma4Family(BaseFamily):
	name = "gemma4"
	# Gemma 4's processor needs nested per-sample image lists so it can pair
	# each image with the correct text when expanding <image> placeholders.
	flatten_vision_inputs = False

	# Soft token budgets supported by Gemma 4's visual tower.
	SUPPORTED_BUDGETS = (70, 140, 280, 560, 1120)

	def decode_skip_special_tokens(self):
		# Keep channel markers visible so parse_response can split thinking from answer.
		return False

	def build_content_block(self, is_video, pil_image, pil_frames, prompt_text):
		# Gemma's processor sub-samples frames under {"type": "video"} (default num_frames=32),
		# which fails if we pre-extract fewer frames. Pass each frame as a separate image instead.
		content = []
		if is_video:
			for frame in pil_frames or []:
				content.append({"type": "image", "image": frame})
		else:
			content.append({"type": "image", "image": pil_image})
		content.append({"type": "text", "text": prompt_text})
		return content

	def processor_kwargs(self, max_resolution=768, vision_token_budget=None):
		kwargs = {}
		if vision_token_budget and vision_token_budget in self.SUPPORTED_BUDGETS:
			kwargs["visual_token_budget"] = vision_token_budget
		return kwargs

	def apply_template(self, processor, messages):
		sig_params = ()
		try:
			sig_params = tuple(inspect.signature(processor.apply_chat_template).parameters.keys())
		except (TypeError, ValueError):
			pass
		kwargs = {"tokenize": False, "add_generation_prompt": True}
		if "enable_thinking" in sig_params:
			kwargs["enable_thinking"] = False
		return processor.apply_chat_template(messages, **kwargs)

	def clean_output(self, processor, decoded_text):
		if hasattr(processor, "parse_response"):
			try:
				parsed = processor.parse_response(decoded_text)
				if isinstance(parsed, dict):
					for key in ("response", "answer", "final", "content", "text"):
						if key in parsed and isinstance(parsed[key], str):
							return parsed[key]
				elif isinstance(parsed, str):
					return parsed
			except Exception:
				pass
		# Fallback: strip thinking channel and Gemma turn markers if parser is unavailable.
		text = decoded_text
		# If a thinking channel is present, keep only the content after it.
		if "thought" in text:
			for sep in ("<channel|>", "<|channel|>", "</channel>"):
				if sep in text:
					text = text.rsplit(sep, 1)[-1]
					break
		import re
		text = re.sub(r"<\|?[a-zA-Z_/][^>]*\|?>", "", text)
		text = text.replace("<bos>", "").replace("<eos>", "")
		return text


def _try_apply_kwargs(func, kwargs):
	"""Filter kwargs to those accepted by func. Returns the filtered dict."""
	try:
		params = inspect.signature(func).parameters
	except (TypeError, ValueError):
		return kwargs
	if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
		return kwargs
	return {k: v for k, v in kwargs.items() if k in params}


def get_family(probe_info, model_path=""):
	backend = (probe_info or {}).get("backend", "")
	arch = (probe_info or {}).get("architecture", "")
	lower_path = (model_path or "").lower()

	if backend == "gemma_hf" or arch.startswith("gemma4") or "gemma-4" in lower_path or "gemma4" in lower_path:
		return Gemma4Family()
	return QwenFamily()
