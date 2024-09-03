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


@register_validator(
    name="guardrails/nsfw_text", data_type="string", has_guardrails_endpoint=True
)
class NSFWText(Validator):
    """Validates that the generated text is safe for work (SFW).

    **Key Properties**
    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `guardrails/nsfw_text`            |
    | Supported data types          | `string`                          |
    | Programmatic fix              | None                              |

    Args:
        threshold: The confidence threshold over which model inferences are considered.
            Defaults to 0.8.
        validation_method: Whether to validate at the sentence level or
            over the full text. Must be one of `sentence` or `full`.
            Defaults to `sentence`.

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
        device: Optional[Union[str, int]] = "cpu",
        model_name: Optional[str] = "michellejieli/NSFW_text_classifier",
        on_fail: Optional[Callable[..., Any]] = None,
        **kwargs,
    ):
        super().__init__(
            on_fail=on_fail,
            threshold=threshold,
            validation_method=validation_method,
            **kwargs,
        )
        self._threshold = float(threshold)
        if validation_method not in ["sentence", "full"]:
            raise ValueError("validation_method must be 'sentence' or 'full'.")
        self._validation_method = validation_method
        if self.use_local:
            self._model = pipeline(
                "text-classification",
                model=model_name,
            )

    def get_nsfw(self, value: str) -> List[str]:
        """Check whether the generated text is NSFW.

        Returns the labels predicted by the model with
        confidence higher than the threshold.

        Args:
            value (str): The generated text.

        Returns:
            pred_labels (List[str]): Labels predicted by the model
            with confidence higher than the threshold.
        """
        pred_labels = []
        if value:
            results = self._model(value)
            if results:
                results = cast(List[List[Dict[str, Any]]], results)
                for result in results:
                    label, score = result["label"], result["score"]
                    if label == "NSFW" and score > self._threshold:
                        pred_labels.append(label)
        return pred_labels

    def validate_each_sentence(
        self, value: str, metadata: Dict[str, Any]
    ) -> ValidationResult:
        sentences = nltk.sent_tokenize(value)

        unsupported_sentences, supported_sentences = [], []
        error_spans: List[ErrorSpan] = []
        char_index = 0

        sentence_predictions = self._inference(sentences)

        for idx, sentence in enumerate(sentences):
            pred_labels = sentence_predictions[idx]

            if pred_labels:
                unsupported_sentences.append(sentence)
                error_spans.append(
                    ErrorSpan(
                        start=char_index,
                        end=char_index + len(sentence),
                        reason=f"NSFW content detected: {', '.join(pred_labels)}",
                    )
                )
            else:
                supported_sentences.append(sentence)
            char_index += len(sentence) + 1  

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

    def validate(self, value: str, metadata: Dict[str, Any]) -> ValidationResult:
        """Validation method for the NSFW text validator."""
        if not value:
            raise ValueError("Value cannot be empty.")

        return self.validate_each_sentence(value, metadata)

    def _inference_local(self, value: str | list) -> List[List[str]]:
        """Local inference method for the NSFW text validator."""

        if isinstance(value, str):
            value = [value]
        predictions = []
        for text in value:
            pred_labels = self.get_nsfw(text)
            predictions.append(pred_labels)

        return predictions

    def _inference_remote(self, value: str | list) -> List[List[str]]:
        """Remote inference method for the NSFW text validator."""

        if isinstance(value, str):
            value = [value]

        request_body = {
            "inputs": [
                {
                    "name": "text",
                    "shape": [len(value)],
                    "data": value,
                    "datatype": "BYTES"
                },
                {
                    "name": "threshold",
                    "shape": [1],
                    "data": [self._threshold],
                    "datatype": "FP32"
                }
            ]
        }
        response = self._hub_inference_request(json.dumps(request_body), self.validation_endpoint)
        if not response or "outputs" not in response:
            raise ValueError("Invalid response from remote inference", response)

        data = [output["data"][0] for output in response["outputs"]]
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
