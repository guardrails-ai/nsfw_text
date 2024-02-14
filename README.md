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

`__init__`

- `on_fail`: The policy to enact when a validator fails.
