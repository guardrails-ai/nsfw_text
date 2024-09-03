import difflib
import json
from typing import Any, Callable, Dict, List, Optional, Union, cast

import nltk
from guardrails.validator_base import (
    ErrorSpan,
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
from transformers import pipeline


@register_validator(name="guardrails/nsfw_text", data_type="string", has_guardrails_endpoint=True)
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
    the prediction is deemed NSFW and return an empty string.
    """

    def __init__(
        self,
        threshold: float = 0.8,
        validation_method: str = "sentence",
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

        if self.use_local:
            self._model = pipeline(
                "text-classification",
                model="michellejieli/NSFW_text_classifier",
            )

    def is_nsfw(self, value: str) -> bool:
        """Determines if the generated text is NSFW.

        Args:
            value (str): The generated text.

        Returns:
            bool: Whether the generated text is NSFW.
        """
        result = self._model(value)
        if not result:
            raise RuntimeError("Failed to get model prediction.")

        pred_label, confidence = result[0]["label"], result[0]["score"]  # type: ignore
        return pred_label == "NSFW" and confidence > self._threshold

    def validate_each_sentence(
        self, value: str, metadata: Dict[str, Any]
    ) -> ValidationResult:
        """Validate that each sentence in the generated text is SFW."""

        sentences = nltk.sent_tokenize(value)

        unsupported_sentences, supported_sentences = [], []
        error_spans: List[ErrorSpan] = []
        char_index = 0

        sentence_predictions = self._inference(sentences)

        for idx, sentence in enumerate(sentences):
            if sentence_predictions[idx]:
                unsupported_sentences.append(sentence)
                error_spans.append(
                    ErrorSpan(
                        start=char_index,
                        end=char_index + len(sentence),
                        reason="NSFW content detected",
                    )
                )
            else:
                supported_sentences.append(sentence)
            char_index += len(sentence) + 1  # Account for the space between sentences

        if unsupported_sentences:
            unsupported_sentences_text = "- " + "\n- ".join(unsupported_sentences)

            return FailResult(
                metadata=metadata,
                error_message=(
                    f"The following sentences in your response "
                    "were found to be NSFW:\n"
                    f"\n{unsupported_sentences_text}"
                ),
                fix_value="\n".join(supported_sentences),
                error_spans=error_spans,
            )
        return PassResult(metadata=metadata)

    def validate_full_text(
        self, value: str, metadata: Dict[str, Any]
    ) -> ValidationResult:
        """Validate that the entire generated text is SFW."""

        if self._inference([value])[0]:
            return FailResult(
                metadata=metadata,
                error_message="The generated text was found to be NSFW.",
            )
        return PassResult(metadata=metadata)

    def validate(self, value: str, metadata: Dict[str, Any]) -> ValidationResult:
        """Validation method of the NSFWText validator."""
        if not value:
            raise ValueError("Value cannot be empty.")

        if self._validation_method == "sentence":
            return self.validate_each_sentence(value, metadata)
        return self.validate_full_text(value, metadata)

    def _inference_local(self, value: str | list) -> List[bool]:
        """Local inference method for the NSFW text validator."""

        if isinstance(value, str):
            value = [value]
        predictions = []
        for text in value:
            predictions.append(self.is_nsfw(text))
        return predictions

    def _inference_remote(self, value: str | list) -> List[bool]:
        """Remote inference method for the NSFW text validator."""

        if isinstance(value, str):
            value = [value]

        request_body = {
            "inputs": [
                {
                    "name": "text",
                    "shape": [len(value)],
                    "data": value,
                    "datatype": "BYTES",
                },
                {
                    "name": "threshold",
                    "shape": [1],
                    "data": [self._threshold],
                    "datatype": "FP32",
                },
            ]
        }
        response = self._hub_inference_request(
            json.dumps(request_body), self.validation_endpoint
        )
        if not response or "outputs" not in response:
            raise ValueError("Invalid response from remote inference", response)

        data = [bool(output["data"][0]) for output in response["outputs"]]
        return data

    def get_error_spans(self, original: str, fixed: str) -> List[ErrorSpan]:
        """Generate error spans to display in failresult (if they exist). Error
        spans show the character-level range of text that has failed validation.

        Args:
            original (str): The input string
            fixed (str): The 'validated' output string

        Returns:
            List[ErrorSpan]: A list of ErrorSpans to represent validation failures
            over the character sequence.
        """
        differ = difflib.Differ()
        diffs = list(differ.compare(original, fixed))
        error_spans = []
        start = None
        for i, diff in enumerate(diffs):
            if diff.startswith("- "):
                if start is None:
                    start = i
            else:
                if start is not None:
                    error_spans.append(
                        ErrorSpan(
                            start=start,
                            end=i,
                            reason="NSFW content detected",
                        )
                    )
                    start = None
        return error_spans
