"""
Translation Service Module
Provides multilingual text translation using NLLB (No Language Left Behind) model.

Supports 200+ languages with high-quality neural machine translation.
"""

import torch
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class TranslationResult:
    """Result from translation."""
    source_text: str
    source_language: str
    target_language: str
    translated_text: str


class TranslationService:
    """
    Multilingual translation using Facebook's NLLB model.

    Uses the distilled 600M parameter version for balance of quality and speed.
    Supports 200+ languages.
    """

    MODEL_ID = "facebook/nllb-200-distilled-600M"

    # Common language codes (NLLB uses FLORES-200 codes)
    LANGUAGE_CODES = {
        "en": "eng_Latn",      # English
        "es": "spa_Latn",      # Spanish
        "fr": "fra_Latn",      # French
        "de": "deu_Latn",      # German
        "it": "ita_Latn",      # Italian
        "pt": "por_Latn",      # Portuguese
        "ru": "rus_Cyrl",      # Russian
        "zh": "zho_Hans",      # Chinese (Simplified)
        "zh-TW": "zho_Hant",   # Chinese (Traditional)
        "ja": "jpn_Jpan",      # Japanese
        "ko": "kor_Hang",      # Korean
        "ar": "arb_Arab",      # Arabic
        "hi": "hin_Deva",      # Hindi
        "bn": "ben_Beng",      # Bengali
        "vi": "vie_Latn",      # Vietnamese
        "th": "tha_Thai",      # Thai
        "tr": "tur_Latn",      # Turkish
        "pl": "pol_Latn",      # Polish
        "nl": "nld_Latn",      # Dutch
        "sv": "swe_Latn",      # Swedish
        "da": "dan_Latn",      # Danish
        "fi": "fin_Latn",      # Finnish
        "no": "nob_Latn",      # Norwegian
        "el": "ell_Grek",      # Greek
        "he": "heb_Hebr",      # Hebrew
        "id": "ind_Latn",      # Indonesian
        "ms": "zsm_Latn",      # Malay
        "tl": "tgl_Latn",      # Tagalog/Filipino
        "uk": "ukr_Cyrl",      # Ukrainian
        "cs": "ces_Latn",      # Czech
        "ro": "ron_Latn",      # Romanian
        "hu": "hun_Latn",      # Hungarian
        "ta": "tam_Taml",      # Tamil
        "te": "tel_Telu",      # Telugu
        "mr": "mar_Deva",      # Marathi
        "gu": "guj_Gujr",      # Gujarati
        "kn": "kan_Knda",      # Kannada
        "ml": "mal_Mlym",      # Malayalam
        "pa": "pan_Guru",      # Punjabi
        "ur": "urd_Arab",      # Urdu
        "fa": "pes_Arab",      # Persian
        "sw": "swh_Latn",      # Swahili
    }

    # Display names for languages
    LANGUAGE_NAMES = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "ru": "Russian",
        "zh": "Chinese (Simplified)",
        "zh-TW": "Chinese (Traditional)",
        "ja": "Japanese",
        "ko": "Korean",
        "ar": "Arabic",
        "hi": "Hindi",
        "bn": "Bengali",
        "vi": "Vietnamese",
        "th": "Thai",
        "tr": "Turkish",
        "pl": "Polish",
        "nl": "Dutch",
        "sv": "Swedish",
        "da": "Danish",
        "fi": "Finnish",
        "no": "Norwegian",
        "el": "Greek",
        "he": "Hebrew",
        "id": "Indonesian",
        "ms": "Malay",
        "tl": "Filipino",
        "uk": "Ukrainian",
        "cs": "Czech",
        "ro": "Romanian",
        "hu": "Hungarian",
        "ta": "Tamil",
        "te": "Telugu",
        "mr": "Marathi",
        "gu": "Gujarati",
        "kn": "Kannada",
        "ml": "Malayalam",
        "pa": "Punjabi",
        "ur": "Urdu",
        "fa": "Persian",
        "sw": "Swahili",
    }

    # Language flags for UI
    LANGUAGE_FLAGS = {
        "en": "🇬🇧",
        "es": "🇪🇸",
        "fr": "🇫🇷",
        "de": "🇩🇪",
        "it": "🇮🇹",
        "pt": "🇵🇹",
        "ru": "🇷🇺",
        "zh": "🇨🇳",
        "zh-TW": "🇹🇼",
        "ja": "🇯🇵",
        "ko": "🇰🇷",
        "ar": "🇸🇦",
        "hi": "🇮🇳",
        "bn": "🇧🇩",
        "vi": "🇻🇳",
        "th": "🇹🇭",
        "tr": "🇹🇷",
        "pl": "🇵🇱",
        "nl": "🇳🇱",
        "sv": "🇸🇪",
        "da": "🇩🇰",
        "fi": "🇫🇮",
        "no": "🇳🇴",
        "el": "🇬🇷",
        "he": "🇮🇱",
        "id": "🇮🇩",
        "ms": "🇲🇾",
        "tl": "🇵🇭",
        "uk": "🇺🇦",
        "cs": "🇨🇿",
        "ro": "🇷🇴",
        "hu": "🇭🇺",
        "ta": "🇮🇳",
        "te": "🇮🇳",
        "mr": "🇮🇳",
        "gu": "🇮🇳",
        "kn": "🇮🇳",
        "ml": "🇮🇳",
        "pa": "🇮🇳",
        "ur": "🇵🇰",
        "fa": "🇮🇷",
        "sw": "🇰🇪",
    }

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self._model = None
        self._tokenizer = None

    def load_model(self):
        """Load the NLLB translation model."""
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        print(f"Loading translation model: {self.MODEL_ID}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.MODEL_ID)
        self._model = self._model.to(self.device)
        print(f"Translation model loaded on {self.device}")

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_length: int = 512
    ) -> TranslationResult:
        """
        Translate text from source language to target language.

        Args:
            text: Text to translate
            source_lang: Source language code (e.g., 'en', 'zh')
            target_lang: Target language code (e.g., 'es', 'fr')
            max_length: Maximum output length

        Returns:
            TranslationResult with translated text
        """
        if self._model is None:
            self.load_model()

        # Map to NLLB language codes
        src_code = self.LANGUAGE_CODES.get(source_lang, "eng_Latn")
        tgt_code = self.LANGUAGE_CODES.get(target_lang, "eng_Latn")

        # Set source language for tokenizer
        self._tokenizer.src_lang = src_code

        # Tokenize
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)

        # Generate translation
        with torch.no_grad():
            generated_tokens = self._model.generate(
                **inputs,
                forced_bos_token_id=self._tokenizer.convert_tokens_to_ids(tgt_code),
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )

        # Decode
        translated_text = self._tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )[0]

        return TranslationResult(
            source_text=text,
            source_language=source_lang,
            target_language=target_lang,
            translated_text=translated_text
        )

    def translate_batch(
        self,
        text: str,
        source_lang: str,
        target_langs: List[str]
    ) -> Dict[str, str]:
        """
        Translate text to multiple target languages.

        Args:
            text: Text to translate
            source_lang: Source language code
            target_langs: List of target language codes

        Returns:
            Dictionary mapping language codes to translated text
        """
        results = {}
        for target_lang in target_langs:
            if target_lang == source_lang:
                results[target_lang] = text
            else:
                result = self.translate(text, source_lang, target_lang)
                results[target_lang] = result.translated_text
        return results

    def get_supported_languages(self) -> List[Dict]:
        """Get list of supported languages with metadata."""
        languages = []
        for code, name in self.LANGUAGE_NAMES.items():
            languages.append({
                "code": code,
                "name": name,
                "flag": self.LANGUAGE_FLAGS.get(code, "🏳️"),
                "nllb_code": self.LANGUAGE_CODES.get(code, ""),
            })
        return sorted(languages, key=lambda x: x["name"])

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None


# Singleton instance
_translator_instance: Optional[TranslationService] = None


def get_translator(use_gpu: bool = False) -> TranslationService:
    """Get or create singleton translator instance."""
    global _translator_instance
    if _translator_instance is None:
        _translator_instance = TranslationService(use_gpu=use_gpu)
    return _translator_instance
