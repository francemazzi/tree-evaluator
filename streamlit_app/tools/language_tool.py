from __future__ import annotations

from typing import Any, Literal, Optional, Type

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

try:
    from langdetect import detect, DetectorFactory
    from langdetect.lang_detect_exception import LangDetectException
    
    # Set seed for reproducibility
    DetectorFactory.seed = 0
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


class LanguageDetectionInput(BaseModel):
    """Input schema for language detection tool."""

    text: str = Field(description="Text to detect the language from")
    

class LanguageTranslationInput(BaseModel):
    """Input schema for language translation tool."""

    text: str = Field(description="Text to translate")
    target_language: Literal["en", "it"] = Field(
        description="Target language code: 'en' for English, 'it' for Italian"
    )
    source_language: Optional[Literal["en", "it"]] = Field(
        default=None,
        description="Source language code (optional, will be auto-detected if not provided): 'en' for English, 'it' for Italian"
    )


class LanguageDetectionTool(BaseTool):
    """Tool to detect the language of user input text."""

    name: str = "detect_language"
    description: str = """
    Detect the language of a given text.
    
    Supports detection of:
    - Italian (it)
    - English (en)
    
    Returns the detected language code and confidence level.
    Use this tool when you need to determine what language the user is speaking.
    """
    args_schema: Type[BaseModel] = LanguageDetectionInput

    _llm: Any = None

    def __init__(self, llm: Any = None, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "_llm", llm)

    def _run(self, text: str) -> dict:
        """Detect the language of the input text."""
        if not text or not text.strip():
            return {
                "detected_language": "unknown",
                "confidence": 0.0,
                "error": "Empty text provided"
            }

        # Try using langdetect first if available
        if LANGDETECT_AVAILABLE:
            try:
                detected_lang = detect(text)
                # Map langdetect codes to our supported languages
                if detected_lang == "it":
                    return {
                        "detected_language": "it",
                        "language_name": "Italian",
                        "confidence": 0.9  # langdetect doesn't provide confidence, use default
                    }
                elif detected_lang == "en":
                    return {
                        "detected_language": "en",
                        "language_name": "English",
                        "confidence": 0.9
                    }
                else:
                    # If detected language is not it or en, try LLM fallback
                    return self._detect_with_llm(text)
            except LangDetectException:
                # Fallback to LLM if langdetect fails
                return self._detect_with_llm(text)
        else:
            # Use LLM if langdetect is not available
            return self._detect_with_llm(text)

    def _detect_with_llm(self, text: str) -> dict:
        """Fallback method using LLM to detect language."""
        if not self._llm:
            return {
                "detected_language": "unknown",
                "confidence": 0.0,
                "error": "LLM not available for language detection"
            }

        prompt = f"""Analyze the following text and determine if it is written in Italian or English.
Return only the language code: "it" for Italian or "en" for English.

Text: {text[:500]}

Language code:"""

        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            detected_lang = response.content.strip().lower()
            
            if detected_lang in ["it", "italian", "italiano"]:
                return {
                    "detected_language": "it",
                    "language_name": "Italian",
                    "confidence": 0.8,
                    "method": "llm"
                }
            elif detected_lang in ["en", "english", "inglese"]:
                return {
                    "detected_language": "en",
                    "language_name": "English",
                    "confidence": 0.8,
                    "method": "llm"
                }
            else:
                return {
                    "detected_language": "unknown",
                    "confidence": 0.0,
                    "error": f"Unexpected LLM response: {detected_lang}"
                }
        except Exception as e:
            return {
                "detected_language": "unknown",
                "confidence": 0.0,
                "error": f"LLM detection failed: {str(e)}"
            }


class LanguageTranslationTool(BaseTool):
    """Tool to translate text between Italian and English."""

    name: str = "translate_text"
    description: str = """
    Translate text between Italian and English.
    
    Supports translation between:
    - Italian (it)
    - English (en)
    
    The source language can be auto-detected if not provided.
    Use this tool when you need to translate user messages or responses.
    """
    args_schema: Type[BaseModel] = LanguageTranslationInput

    _llm: Any = None

    def __init__(self, llm: Any = None, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, "_llm", llm)

    def _run(
        self,
        text: str,
        target_language: Literal["en", "it"],
        source_language: Optional[Literal["en", "it"]] = None
    ) -> dict:
        """Translate text to the target language."""
        if not text or not text.strip():
            return {
                "translated_text": "",
                "source_language": source_language or "unknown",
                "target_language": target_language,
                "error": "Empty text provided"
            }

        if not self._llm:
            return {
                "translated_text": text,
                "source_language": source_language or "unknown",
                "target_language": target_language,
                "error": "LLM not available for translation"
            }

        # If source language is not provided, try to detect it
        if source_language is None:
            detection_tool = LanguageDetectionTool(llm=self._llm)
            detection_result = detection_tool._run(text)
            source_language = detection_result.get("detected_language", "unknown")
            if source_language not in ["en", "it"]:
                return {
                    "translated_text": text,
                    "source_language": "unknown",
                    "target_language": target_language,
                    "error": f"Could not detect source language. Detected: {source_language}"
                }

        # If source and target are the same, return original text
        if source_language == target_language:
            return {
                "translated_text": text,
                "source_language": source_language,
                "target_language": target_language,
                "note": "Source and target languages are the same"
            }

        # Translate using LLM
        target_lang_name = "Italian" if target_language == "it" else "English"
        source_lang_name = "Italian" if source_language == "it" else "English"

        prompt = f"""Translate the following text from {source_lang_name} to {target_lang_name}.
Provide only the translation, without any additional explanation or notes.

Original text ({source_lang_name}):
{text}

Translation ({target_lang_name}):"""

        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            translated_text = response.content.strip()
            
            return {
                "translated_text": translated_text,
                "source_language": source_language,
                "target_language": target_language,
                "source_language_name": source_lang_name,
                "target_language_name": target_lang_name
            }
        except Exception as e:
            return {
                "translated_text": text,
                "source_language": source_language,
                "target_language": target_language,
                "error": f"Translation failed: {str(e)}"
            }

