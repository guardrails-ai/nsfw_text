## Overview

| Developed by | Guardrails AI |
| --- | --- |
| Date of development | Feb 15, 2024 |
| Validator type | Format |
| Blog |  |
| License | Apache 2 |
| Input/Output | Output |

## Description

This validator checks if an LLM-generated text is not safe for work (NSFW). It validates either sentence-by-sentence or the entire text.

## Requirements
- Dependencies: `nltk`, `transformers`

## Installation

```bash
$ guardrails hub install hub://guardrails/nsfw_text
```

## Usage Examples

### Validating string output via Python

In this example, we use the `nsfw_text` validator on any LLM generated text.

```python
# Import Guard and Validator
from guardrails.hub import NSFWText
from guardrails import Guard

# Initialize Validator
val = NSFWText()

# Setup Guard
guard = Guard.from_string(
    validators=[val, ...],
)

# Pass LLM output through guard
guard.parse("Meditation is a good way to relax and enjoy life.")  # Pass
guard.parse("Pissing all over the floor is a good hygiene practice.")  # Fail

```
### Validating JSON output via Python

In this example, we use the `nsfw_text` validator on a pet description string.

```python
# Import Guard and Validator
from pydantic import BaseModel
from guardrails.hub import NSFWText
from guardrails import Guard

val = NSFWText()

# Create Pydantic BaseModel
class PetInfo(BaseModel):
    pet_description: str = Field(validators=[val])

# Create a Guard to check for valid Pydantic output
guard = Guard.from_pydantic(output_class=PetInfo)

# Run LLM output generating JSON through guard
guard.parse("""
{
    "pet_description": "Caesar is a great cat who is fun to hang out with.",
}
""")

guard.parse("""
{
    "pet_description": "Caeser loves to piss all over the floor."
}
""")
```

## API Reference

**`__init__(self, threshold=0.8, validation_method="sentence", on_fail="noop")`**
<ul>

Initializes a new instance of the Validator class.

**Parameters:**

- **threshold** *(float):* The confidence threshold over which model inferences are considered. Default is 0.8.
- **validation_method** *(str):* The method to use for validation. If `sentence`, the validator will validate each sentence in the input text. If `full`, the validator will validate the entire input text. Default is `sentence`.
- **`on_fail`** *(str, Callable):* The policy to enact when a validator fails. If `str`, must be one of `reask`, `fix`, `filter`, `refrain`, `noop`, `exception` or `fix_reask`. Otherwise, must be a function that is called when the validator fails.

</ul>

<br>

**`__call__(self, value, metadata={}) → ValidationOutcome`**

<ul>

Validates the given `value` using the rules defined in this validator, relying on the `metadata` provided to customize the validation process. This method is automatically invoked by `guard.parse(...)`, ensuring the validation logic is applied to the input data.

Note:

1. This method should not be called directly by the user. Instead, invoke `guard.parse(...)` where this method will be called internally for each associated Validator.
2. When invoking `guard.parse(...)`, ensure to pass the appropriate `metadata` dictionary that includes keys and values required by this validator. If `guard` is associated with multiple validators, combine all necessary metadata into a single dictionary.

**Parameters:**

- **`value`** *(Any):* The input value to validate.
- **`metadata`** *(dict):* A dictionary containing metadata required for validation. No additional metadata keys are needed for this validator.

</ul>
