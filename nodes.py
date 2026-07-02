"""Text Encode (Krea2) — vision-aware conditioning for the Krea2 / K2 DiT.

Krea2 conditions on a 12-layer Qwen3-VL-4B tap (see ``comfy/text_encoders/krea2.py``).
Because that text encoder is a vision-language model, a reference image can be fed through
its *vision* path so the conditioning becomes visually informed by the image — without any
VAE / reference-latent. The Krea2 DiT (``comfy/ldm/krea2/model.py``) is pure text-to-image:
its sequence is ``[text_tokens, noisy_image_patches]`` with no slot for a reference latent,
so a VAE input would be a no-op here and is deliberately omitted.

Each reference image has an optional companion mask. When a mask is connected the image is
cropped to the mask's bounding box before the vision encoder, so the VLM only "sees" the
masked region. (This is reference-image masking; it is not inpainting — Krea2 has no
inpaint/concat pathway to regenerate a masked output region.)

This node differs from ``TextEncodeQwenImageEdit`` in two ways:
  * it forces the Krea2 *descriptor* conditioning template even when images are attached
    (the core Qwen-Edit node falls back to Qwen3-VL's plain image template), and
  * it has no VAE input, and it accepts an unbounded, auto-growing set of image+mask slots.
"""

import json
import logging
import math
import os
import re
import hashlib
from collections import OrderedDict

import torch

import comfy.utils

LOGGER = logging.getLogger(__name__)

# Keep in sync with the model's own template; fall back to a literal copy on non-Krea2 builds.
try:
    from comfy.text_encoders.krea2 import KREA2_TEMPLATE
except Exception:  # pragma: no cover - portability shim
    KREA2_TEMPLATE = (
        "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, "
        "quantity, text, spatial relationships of the objects and background:<|im_end|>\n"
        "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    )

# The user-facing system_prompt field holds just the system *message text*; the node wraps it in
# the chat-template scaffolding. Pull the default (Krea2's trained descriptor) out of the template
# so it stays in sync with whatever comfy ships.
_sys = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", KREA2_TEMPLATE, re.S)
KREA2_SYSTEM_DEFAULT = _sys.group(1) if _sys else (
    "Describe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:"
)

# Instruct/edit-style framing (à la TextEncodeQwenImageEditPlus): paste this into system_prompt to
# make the VLM fuse the user's text WITH the reference image instead of just describing it.
# Out-of-distribution for Krea2's trained descriptor — experimental.
KREA2_INSTRUCT_SYSTEM = (
    "Describe the key features of the reference image (color, shape, size, texture, objects, "
    "background), then explain how the user's instruction should combine with or alter it, and "
    "generate a new image meeting the instruction while staying consistent with the reference "
    "where appropriate:"
)


KREA2_JSON_COMPACT_CHARS = int(os.environ.get("KREA2_TEXTENCODER_JSON_COMPACT_CHARS", "2600"))
KREA2_TEXTENCODER_CACHE_SIZE = int(os.environ.get("KREA2_TEXTENCODER_CACHE_SIZE", "4"))
KREA2_JSON_PROMPT_MODES = ("json_structured", "json_minify", "json_minify_or_prose", "prose_compact")
KREA2_JSON_PROMPT_MODE = os.environ.get("KREA2_TEXTENCODER_JSON_MODE", "json_structured").strip().lower()
if KREA2_JSON_PROMPT_MODE not in KREA2_JSON_PROMPT_MODES:
    KREA2_JSON_PROMPT_MODE = "json_structured"
KREA2_JSON_TOP_LEVEL_HINTS = {
    "subject",
    "hair",
    "body",
    "pose",
    "clothing",
    "accessories",
    "photography",
    "background",
    "the_vibe",
    "constraints",
    "negative_prompt",
}
KREA2_JSON_SECTION_ORDER = (
    "constraints",
    "background",
    "photography",
    "subject",
    "pose",
    "body",
    "hair",
    "clothing",
    "accessories",
    "the_vibe",
)
KREA2_JSON_SKIP_STRINGS = KREA2_JSON_TOP_LEVEL_HINTS | {
    "description",
    "mirror_rules",
    "age",
    "expression",
    "eyes",
    "look",
    "energy",
    "direction",
    "mouth",
    "position",
    "overall",
    "face",
    "preserve_original",
    "makeup",
    "color",
    "style",
    "effect",
    "frame",
    "waist",
    "chest",
    "legs",
    "skin",
    "visible_areas",
    "tone",
    "texture",
    "lighting_effect",
    "base",
    "top",
    "bottom",
    "type",
    "details",
    "headwear",
    "jewelry",
    "device",
    "prop",
    "camera_style",
    "angle",
    "shot_type",
    "aspect_ratio",
    "lighting",
    "depth_of_field",
    "composition",
    "crop_control",
    "lens",
    "setting",
    "wall_color",
    "elements",
    "atmosphere",
    "mood",
    "aesthetic",
    "authenticity",
    "intimacy",
    "story",
    "caption_energy",
    "must_keep",
    "avoid",
}


def _clean_prompt_leaf(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return re.sub(r"\s+", " ", str(value)).strip(" ,;")


def _prompt_leaves(value):
    if isinstance(value, dict):
        leaves = []
        for item in value.values():
            leaves.extend(_prompt_leaves(item))
        return leaves
    if isinstance(value, list):
        leaves = []
        for item in value:
            leaves.extend(_prompt_leaves(item))
        return leaves
    text = _clean_prompt_leaf(value)
    return [text] if text else []


def _prompt_at_path(value, *path):
    current = value
    for part in path:
        if not isinstance(current, dict):
            return []
        current = current.get(part)
    return _prompt_leaves(current)


def _dedupe_items(items, seen, limit):
    out = []
    for item in items:
        text = _clean_prompt_leaf(item)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _append_compact_segment(segments, seen, label, items, limit=12):
    values = _dedupe_items(items, seen, limit)
    if values:
        segments.append(f"{label}: " + "; ".join(values))


def _truncate_prompt(text, max_chars):
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars].rstrip()
    boundary = max(clipped.rfind(". "), clipped.rfind("; "), clipped.rfind(", "))
    if boundary >= max(240, int(max_chars * 0.7)):
        clipped = clipped[: boundary + 1].rstrip()
    return clipped.rstrip(" ,;.")


def _join_compact_segments(segments, max_chars):
    result = []
    for segment in segments:
        candidate = ". ".join([*result, segment]) if result else segment
        if len(candidate) <= max_chars:
            result.append(segment)
        elif not result:
            return _truncate_prompt(segment, max_chars)
    return ". ".join(result) if result else ""


def _looks_like_krea2_json_prompt(text):
    if len(text) < 200:
        return False
    lowered = text.lower()
    return sum(1 for key in KREA2_JSON_TOP_LEVEL_HINTS if f'"{key}"' in lowered or key in lowered) >= 4


JSON_MISSING_COMMA_BEFORE_PROPERTY_PATTERN = re.compile(
    r'((?:true|false|null)|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|[}\]"])'
    r'([ \t\r]*\n[ \t]*)(?="[^"\n]{1,160}"[ \t]*:)'
)
JSON_MISSING_COMMA_BETWEEN_STRING_ITEMS_PATTERN = re.compile(
    r'(")([ \t\r]*\n[ \t]*)(?="[^"\n]*"[ \t\r\n]*(?:,|\]))'
)


def _repair_common_json_delimiters(text):
    repaired = JSON_MISSING_COMMA_BEFORE_PROPERTY_PATTERN.sub(r"\1,\2", text)
    repaired = JSON_MISSING_COMMA_BETWEEN_STRING_ITEMS_PATTERN.sub(r"\1,\2", repaired)
    return repaired


def _loads_json_with_common_repairs(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        repaired = _repair_common_json_delimiters(text)
        if repaired != text:
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass
        raise exc


def _extract_krea2_json_text(text):
    json_text = str(text or "").strip()
    if not json_text.startswith("{"):
        start = json_text.find("{")
        end = json_text.rfind("}")
        if start >= 0 and end > start:
            json_text = json_text[start : end + 1].strip()
    return json_text


def _normalize_krea2_json_like_text(text):
    json_text = _extract_krea2_json_text(text)
    if not _looks_like_krea2_json_prompt(json_text):
        return text
    # Keep the JSON/key/value structure for Krea2, but remove wrapper text and indentation.
    lines = [line.strip() for line in json_text.splitlines() if line.strip()]
    return "\n".join(lines) if lines else json_text.strip()


def _load_krea2_json_prompt_value(prompt):
    text = str(prompt or "").strip()
    if len(text) < 200:
        return None
    json_text = _extract_krea2_json_text(text)
    value = _loads_json_with_common_repairs(json_text)
    if not isinstance(value, dict) or len(KREA2_JSON_TOP_LEVEL_HINTS.intersection(value.keys())) < 4:
        return None
    return value


def _strip_krea2_negation_lists(value):
    # negative_prompt / constraints.avoid are meant for a separate negative-conditioning pass.
    # This compacted text is encoded as POSITIVE conditioning, and the Krea2 turbo workflows
    # run CFG 1.0 where the negative pass is skipped, so keeping the lists only injects the
    # named failure concepts into the image and slows every denoise step.
    if isinstance(value, dict):
        value.pop("negative_prompt", None)
        constraints = value.get("constraints")
        if isinstance(constraints, dict):
            constraints.pop("avoid", None)
    return value


def _minify_krea2_json_prompt(prompt, max_chars=KREA2_JSON_COMPACT_CHARS):
    text = str(prompt or "").strip()
    try:
        value = _load_krea2_json_prompt_value(text)
    except Exception:
        return _normalize_krea2_json_like_text(text)
    if value is None:
        return prompt
    value = _strip_krea2_negation_lists(value)
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _structured_krea2_json_prompt(prompt):
    text = str(prompt or "").strip()
    try:
        value = _load_krea2_json_prompt_value(text)
    except Exception:
        return _normalize_krea2_json_like_text(text)
    if value is None:
        return prompt
    value = _strip_krea2_negation_lists(value)
    return json.dumps(value, ensure_ascii=False, indent=1, separators=(",", ": "))


def _extract_quoted_prompt_values(text):
    values = []
    for match in re.finditer(r'"((?:\\.|[^"\\])*)"', text):
        tail = text[match.end() : match.end() + 12]
        if re.match(r"\s*:", tail):
            continue
        raw = match.group(1)
        try:
            value = json.loads('"' + raw + '"')
        except Exception:
            value = raw.replace(r"\"", '"').replace(r"\\", "\\")
        cleaned = _clean_prompt_leaf(value)
        if not cleaned:
            continue
        if cleaned.lower() in KREA2_JSON_SKIP_STRINGS:
            continue
        values.append(cleaned)
    return values


def _fallback_compact_krea2_json_like_prompt(text, max_chars):
    if not _looks_like_krea2_json_prompt(text):
        return text

    positions = []
    lowered = text.lower()
    for section in KREA2_JSON_SECTION_ORDER:
        index = lowered.find(f'"{section}"')
        if index < 0:
            index = lowered.find(section)
        if index >= 0:
            positions.append((index, section))
    positions.sort()

    segments = []
    seen = set()
    for order_section in KREA2_JSON_SECTION_ORDER:
        for pos_index, (start, section) in enumerate(positions):
            if section != order_section:
                continue
            end = positions[pos_index + 1][0] if pos_index + 1 < len(positions) else len(text)
            values = _extract_quoted_prompt_values(text[start:end])
            _append_compact_segment(segments, seen, section.replace("_", " "), values, limit=16)

    if not segments:
        _append_compact_segment(segments, seen, "prompt", _extract_quoted_prompt_values(text), limit=48)
    compact = _join_compact_segments(segments, max_chars)
    return compact or _truncate_prompt(re.sub(r"\s+", " ", text).strip(), max_chars)


def _prose_compact_krea2_json_prompt(prompt, max_chars=KREA2_JSON_COMPACT_CHARS):
    text = str(prompt or "").strip()
    if len(text) < 200:
        return prompt
    try:
        value = _load_krea2_json_prompt_value(text)
    except Exception:
        return _fallback_compact_krea2_json_like_prompt(text, max_chars)
    if value is None:
        return prompt

    max_chars = max(300, min(8000, int(max_chars or KREA2_JSON_COMPACT_CHARS)))
    segments = []
    seen = set()
    _append_compact_segment(
        segments,
        seen,
        "must keep",
        _prompt_at_path(value, "constraints", "must_keep"),
        limit=24,
    )
    _append_compact_segment(
        segments,
        seen,
        "scene",
        [
            *_prompt_at_path(value, "background", "setting"),
            *_prompt_at_path(value, "background", "elements"),
            *_prompt_at_path(value, "accessories", "prop"),
        ],
        limit=10,
    )
    _append_compact_segment(
        segments,
        seen,
        "camera and framing",
        [
            *_prompt_at_path(value, "photography", "shot_type"),
            *_prompt_at_path(value, "photography", "composition"),
            *_prompt_at_path(value, "photography", "crop_control"),
            *_prompt_at_path(value, "photography", "angle"),
            *_prompt_at_path(value, "photography", "lens"),
            *_prompt_at_path(value, "photography", "aspect_ratio"),
        ],
        limit=14,
    )
    _append_compact_segment(
        segments,
        seen,
        "subject",
        [
            *_prompt_at_path(value, "subject", "description"),
            *_prompt_at_path(value, "subject", "age"),
            *_prompt_at_path(value, "subject", "mirror_rules"),
            *_prompt_at_path(value, "subject", "face"),
            *_prompt_at_path(value, "subject", "expression"),
        ],
        limit=14,
    )
    _append_compact_segment(
        segments,
        seen,
        "pose and body placement",
        [*_prompt_at_path(value, "pose"), *_prompt_at_path(value, "body")],
        limit=16,
    )
    _append_compact_segment(
        segments,
        seen,
        "appearance",
        [*_prompt_at_path(value, "hair"), *_prompt_at_path(value, "clothing"), *_prompt_at_path(value, "accessories")],
        limit=14,
    )
    # No "avoid" segment: constraints.avoid / negative_prompt are negation lists meant for a
    # separate negative pass; naming those concepts in positive prose summons them instead.
    _append_compact_segment(
        segments,
        seen,
        "style",
        [
            *_prompt_at_path(value, "photography", "camera_style"),
            *_prompt_at_path(value, "photography", "lighting"),
            *_prompt_at_path(value, "photography", "depth_of_field"),
            *_prompt_at_path(value, "photography", "texture"),
            *_prompt_at_path(value, "the_vibe"),
        ],
        limit=12,
    )
    compact = _join_compact_segments(segments, max_chars)
    return compact or prompt


def _compact_krea2_json_prompt(prompt, max_chars=KREA2_JSON_COMPACT_CHARS, mode=KREA2_JSON_PROMPT_MODE):
    normalized_mode = (mode or KREA2_JSON_PROMPT_MODE).strip().lower()
    if normalized_mode not in KREA2_JSON_PROMPT_MODES:
        normalized_mode = "json_structured"

    if normalized_mode == "prose_compact":
        return _prose_compact_krea2_json_prompt(prompt, max_chars=max_chars)

    if normalized_mode == "json_minify_or_prose":
        text = str(prompt or "").strip()
        minified = _minify_krea2_json_prompt(prompt, max_chars=max_chars)
        if isinstance(minified, str) and minified != text and len(minified) <= max_chars:
            return minified
        if not _looks_like_krea2_json_prompt(text):
            return prompt
        return _prose_compact_krea2_json_prompt(prompt, max_chars=max_chars)

    if normalized_mode == "json_minify":
        return _minify_krea2_json_prompt(prompt, max_chars=max_chars)
    return _structured_krea2_json_prompt(prompt)


def _clone_conditioning_value(value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, dict):
        return {key: _clone_conditioning_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_conditioning_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_conditioning_value(item) for item in value)
    return value


def _text_conditioning_cache_key(clip, text, template):
    h = hashlib.sha256()
    h.update(str(id(clip)).encode("ascii", "ignore"))
    h.update(b"\0")
    h.update(str(template or "").encode("utf-8", "surrogatepass"))
    h.update(b"\0")
    h.update(str(text or "").encode("utf-8", "surrogatepass"))
    return h.hexdigest()


class TextEncodeKrea2:
    def __init__(self):
        self._text_conditioning_cache = OrderedDict()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            },
            "optional": {
                # system_prompt sits just above the image slots.
                "system_prompt": ("STRING", {
                    "forceInput": True,
                    "tooltip": "Optional system-instruction input. Wire a text node to override how the "
                               "VLM frames the reference + your prompt; leave unconnected to use Krea2's "
                               "trained descriptor (in-distribution). Use an instruct/edit-style "
                               "instruction (see README) to fuse the prompt with the image. The node "
                               "adds the chat-template scaffolding; provide just the instruction text.",
                }),
                # image1/mask1 are the seed pair; the web extension grows image2/mask2, ... on connect.
                "image1": ("IMAGE",),
                "mask1": ("MASK",),
                "vision_megapixels": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1,
                    "tooltip": "Maximum size (in megapixels) for each reference before the Qwen3-VL "
                               "vision encoder. References larger than this are downscaled; smaller "
                               "ones (e.g. a tight mask crop) are kept at native size, never upscaled.",
                }),
                "mask_padding": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.02,
                    "tooltip": "Context kept around the mask before cropping, as a fraction of the "
                               "image size added on EACH side. 0 = tight crop to the mask; 0.1 = ~10% "
                               "margin of surroundings. Only applies when a mask is connected.",
                }),
                "vision_position": (["before prompt", "after prompt"], {
                    "default": "before prompt",
                    "tooltip": "Where the image (vision) tokens sit in the user turn relative to your "
                               "text. 'before prompt' = image then text (default); 'after prompt' = text "
                               "then image. No effect without an image. Experimental.",
                }),
                "print_prompt": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print the full assembled prompt sent to the Qwen3-VL encoder (system "
                               "instruction + vision placeholders + your text) to the ComfyUI console.",
                }),
                "auto_compact_json": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically optimize Krea2 photo JSON prompts before tokenization. "
                               "The default mode preserves raw JSON line structure for adherence.",
                }),
                "json_prompt_mode": (list(KREA2_JSON_PROMPT_MODES), {
                    "default": KREA2_JSON_PROMPT_MODE,
                    "tooltip": "json_structured is adherence-first. json_minify and prose_compact are "
                               "speed-testing modes and can reduce prompt adherence.",
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "model/conditioning/krea2"
    DESCRIPTION = ("Krea2 (K2) text conditioning with optional vision prompting. Reference images are "
                   "fed through the Qwen3-VL vision path; an optional per-image mask crops the image to "
                   "the masked region. No VAE is used (Krea2 has no reference-latent pathway).")

    @staticmethod
    def _collect_indexed(kwargs, prefix):
        pattern = re.compile(r"^{}(\d+)$".format(prefix))
        out = {}
        for key, value in kwargs.items():
            match = pattern.match(key)
            if match is not None and value is not None:
                out[int(match.group(1))] = value
        return out

    @staticmethod
    def _crop_to_mask(image, mask, padding=0.0):
        """Crop image (B,H,W,C) to the mask bounding box, expanded by `padding` (a
        fraction of the image size) on each side. No-op if mask empty/None."""
        if mask is None:
            return image

        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        elif mask.dim() == 4:  # (B,1,H,W) or similar -> (B,H,W)
            mask = mask.reshape(-1, mask.shape[-2], mask.shape[-1])

        h, w = image.shape[1], image.shape[2]
        if mask.shape[-2:] != (h, w):
            resized = comfy.utils.common_upscale(mask.unsqueeze(1), w, h, "bilinear", "disabled")
            mask = resized[:, 0]

        presence = (mask > 0.5).any(dim=0)  # collapse batch -> (H,W)
        if not bool(presence.any()):
            return image  # nothing selected: keep the whole image

        rows = torch.where(torch.any(presence, dim=1))[0]
        cols = torch.where(torch.any(presence, dim=0))[0]
        y0, y1 = int(rows[0]), int(rows[-1])
        x0, x1 = int(cols[0]), int(cols[-1])

        if padding > 0.0:  # grow the box outward for surrounding context, clamped to the image
            pad_x = round(padding * w)
            pad_y = round(padding * h)
            x0 = max(0, x0 - pad_x)
            x1 = min(w - 1, x1 + pad_x)
            y0 = max(0, y0 - pad_y)
            y1 = min(h - 1, y1 + pad_y)

        return image[:, y0:y1 + 1, x0:x1 + 1, :]

    @classmethod
    def _prepare_vision(cls, kwargs, vision_megapixels, mask_padding):
        """Crop+resize each connected reference and build the vision-token string."""
        images = cls._collect_indexed(kwargs, "image")
        masks = cls._collect_indexed(kwargs, "mask")
        ordered = sorted(images.keys())

        images_vl = []
        image_prompt = ""
        total = int(vision_megapixels * 1024 * 1024)

        for slot, n in enumerate(ordered):
            image = cls._crop_to_mask(images[n], masks.get(n), padding=mask_padding)
            samples = image.movedim(-1, 1)
            # vision_megapixels is an upper CAP, not a fixed target: only downscale oversized
            # references, never upscale (a small mask crop would otherwise be magnified).
            scale_by = min(1.0, math.sqrt(total / (samples.shape[3] * samples.shape[2])))
            width = round(samples.shape[3] * scale_by)
            height = round(samples.shape[2] * scale_by)
            s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
            images_vl.append(s.movedim(1, -1)[:, :, :, :3])
            if len(ordered) > 1:
                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(slot + 1)
            else:
                image_prompt += "<|vision_start|><|image_pad|><|vision_end|>"
        return images_vl, image_prompt

    @staticmethod
    def _build_text(system_prompt, prompt, image_prompt, vision_position):
        """Assemble the user text (with vision tokens) and the chat template."""
        system = system_prompt.strip() or KREA2_SYSTEM_DEFAULT
        template = ("<|im_start|>system\n" + system + "<|im_end|>\n"
                    "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n")
        text = (prompt + image_prompt) if vision_position == "after prompt" else (image_prompt + prompt)
        return text, template

    @staticmethod
    def _fp8_hint(exc, images_vl):
        """Map the cryptic FP8 vision crash to an actionable error; else None.

        ComfyUI's Qwen3-VL vision tower (qwen35.py fast_pos_embed_interpolate) adds the pos-embed
        weights without casting, so an FP8-loaded text encoder dies on the image path."""
        if images_vl and isinstance(exc, NotImplementedError) and "Float8" in str(exc):
            return RuntimeError(
                "Krea2: the Qwen3-VL text encoder is loaded in FP8, which ComfyUI's vision tower "
                "cannot run on the image path ('add_stub not implemented for Float8_e4m3fn'). Load "
                "a bf16/fp16 Qwen3-VL-4B text encoder (e.g. a qwen3vl_4b *bf16* file) via CLIPLoader "
                "type 'krea2' when using image references. The FP8 encoder works only text-only."
            )
        return None

    def encode(self, clip, prompt, vision_megapixels=1.0, mask_padding=0.0,
               system_prompt=KREA2_SYSTEM_DEFAULT, vision_position="before prompt",
               print_prompt=False, auto_compact_json=True,
               json_prompt_mode=KREA2_JSON_PROMPT_MODE, **kwargs):
        images_vl, image_prompt = self._prepare_vision(kwargs, vision_megapixels, mask_padding)
        if auto_compact_json:
            raw_prompt = str(prompt or "")
            optimized_prompt = _compact_krea2_json_prompt(prompt, mode=json_prompt_mode)
            if isinstance(optimized_prompt, str) and optimized_prompt != raw_prompt:
                LOGGER.info(
                    "TextEncodeKrea2 optimized prompt: mode=%s raw_chars=%d optimized_chars=%d references=%d",
                    json_prompt_mode,
                    len(raw_prompt),
                    len(optimized_prompt),
                    len(images_vl),
                )
            prompt = optimized_prompt
        text, template = self._build_text(system_prompt, prompt, image_prompt, vision_position)

        if print_prompt:
            print("\n========== Text Encode (Krea2) -> Qwen3-VL prompt ==========")
            print(template.replace("{}", text, 1))  # literal replace: brace-safe
            print("---- references: {} ----".format(len(images_vl)))
            print("===========================================================\n")

        cache_key = None
        cache_enabled = KREA2_TEXTENCODER_CACHE_SIZE > 0 and not images_vl
        if cache_enabled:
            cache_key = _text_conditioning_cache_key(clip, text, template)
            cached = self._text_conditioning_cache.get(cache_key)
            if cached is not None:
                self._text_conditioning_cache.move_to_end(cache_key)
                LOGGER.info(
                    "TextEncodeKrea2 conditioning cache hit: prompt_chars=%d references=0 key=%s",
                    len(str(prompt or "")),
                    cache_key[:12],
                )
                return (_clone_conditioning_value(cached),)

        tokens = clip.tokenize(text, images=images_vl, llama_template=template)
        try:
            conditioning = clip.encode_from_tokens_scheduled(tokens)
        except NotImplementedError as exc:
            hint = self._fp8_hint(exc, images_vl)
            if hint is not None:
                raise hint from exc
            raise
        if cache_enabled and cache_key is not None:
            self._text_conditioning_cache[cache_key] = _clone_conditioning_value(conditioning)
            self._text_conditioning_cache.move_to_end(cache_key)
            while len(self._text_conditioning_cache) > KREA2_TEXTENCODER_CACHE_SIZE:
                self._text_conditioning_cache.popitem(last=False)
            LOGGER.info(
                "TextEncodeKrea2 conditioning cache store: prompt_chars=%d references=0 key=%s size=%d",
                len(str(prompt or "")),
                cache_key[:12],
                len(self._text_conditioning_cache),
            )
        return (conditioning,)


class Krea2SystemPrompt:
    """Generic text node preloaded with the instruct/edit-style system prompt. Wire its
    output into TextEncodeKrea2's `system_prompt` input to make the prompt fuse with the
    reference image (experimental / out-of-distribution). Edit the text freely."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True, "default": KREA2_INSTRUCT_SYSTEM,
                    "tooltip": "System instruction for Krea2's VLM. Defaults to an instruct/edit-style "
                               "framing that fuses your prompt with the reference image. Edit as needed; "
                               "paste the plain descriptor to fall back to default behavior.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)
    FUNCTION = "run"
    CATEGORY = "model/conditioning/krea2"
    DESCRIPTION = ("Text node preloaded with an instruct-style system prompt for Text Encode (Krea2). "
                   "Wire its output into the encoder's system_prompt input.")

    def run(self, text):
        return (text,)


NODE_CLASS_MAPPINGS = {
    "TextEncodeKrea2": TextEncodeKrea2,
    "Krea2SystemPrompt": Krea2SystemPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextEncodeKrea2": "Text Encode (Krea2)",
    "Krea2SystemPrompt": "Krea2 System Prompt",
}
