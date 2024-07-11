from typing import Any, Callable, Dict, Optional

import nltk
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
from transformers import pipeline


@register_validator(name="guardrails/nsfw_text", data_type="string")
class NSFWText(Validator):
    """Validates that the generated text is safe for work (SFW).

    **Key Properties**
    | Property                      | Description            |
    | ----------------------------- | ---------------------- |
    | Name for `format` attribute   | `guardrails/nsfw_text` |
    | Supported data types          | `string`               |
    | Programmatic fix              | N/A                    |

    Args:
        threshold: The confidence threshold over which model inferences are considered.
            Must be a float between 0 and 1. Defaults to 0.8
        validation_method: Whether to validate at the sentence level or
            over the full text. Must be one of `sentence` or `full`.
            Defaults to `sentence`

    This validator uses the pre-trained multi-class model from HuggingFace -
    `michellejieli/NSFW_text_classifier` to check whether the generated text is
    safe for work. If the model predicts the text to be "NSFW" with a confidence
    higher than the threshold, the validator fails. Otherwise, it passes.

    If validation_method is `sentence`, the validator will remove the sentences
    that are predicted to be NSFW and return the remaining sentences. If
    validation_method is `full`, the validator will remove the entire text if
    the prediction is deemed NSFW it will return an empty string.
    """

    def __init__(
        self,
        threshold: float = 0.8,
        validation_method: str = "sentence",
        device="cpu",
        on_fail: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(
            on_fail, threshold=threshold, validation_method=validation_method, **kwargs
        )
        self._threshold = float(threshold)
        if validation_method not in ["sentence", "full"]:
            raise ValueError("validation_method must be 'sentence' or 'full'.")
        self._validation_method = validation_method

        # Define the model, pipeline and labels
        self._model_name = "michellejieli/NSFW_text_classifier"
        self._pipe = pipeline(
            "text-classification",
            model=self._model_name,
            device=device,
        )
        print("Pipeline setup successfully.")

    def is_nsfw(self, value: str) -> bool:
        """Determines if the generated text is NSFW.

        Args:
            value (str): The generated text.

        Returns:
            bool: Whether the generated text is NSFW.
        """
        result = self._pipe(value)
        if not result:
            raise RuntimeError("Failed to get model prediction.")

        pred_label, confidence = result[0]["label"], result[0]["score"]  # type: ignore
        if pred_label == "NSFW" and confidence > self._threshold:
            return True
        return False

    def validate_each_sentence(
        self, value: str, metadata: Dict[str, Any]
    ) -> ValidationResult:
        """Validate that each sentence in the generated text is SFW."""

        # Split the value into sentences using nltk sentence tokenizer.
        sentences = nltk.sent_tokenize(value)

        unsupported_sentences, supported_sentences = [], []
        for sentence in sentences:
            is_nsfw = self.is_nsfw(sentence)
            if is_nsfw:
                unsupported_sentences.append(sentence)
            else:
                supported_sentences.append(sentence)

        if unsupported_sentences:
            unsupported_sentences = "- " + "\n- ".join(unsupported_sentences)
            return FailResult(
                metadata=metadata,
                error_message=(
                    f"The following sentences in your response "
                    "were found to be NSFW:\n"
                    f"\n{unsupported_sentences}"
                ),
                fix_value="\n".join(supported_sentences),
            )
        return PassResult()

    def validate_full_text(
        self, value: str, metadata: Dict[str, Any]
    ) -> ValidationResult:
        """Validate that the entire generated text is SFW."""

        is_nsfw = self.is_nsfw(value)
        if is_nsfw:
            return FailResult(
                metadata=metadata,
                error_message="The generated text was found to be NSFW.",
            )
        return PassResult()

    def validate(self, value: str, metadata: Dict[str, Any]) -> ValidationResult:
        """Validation method of the NSFWText validator."""
        if self._validation_method == "sentence":
            return self.validate_each_sentence(value, metadata)

        if self._validation_method == "full":
            return self.validate_full_text(value, metadata)
