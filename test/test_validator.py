from guardrails import Guard
from pydantic import BaseModel, Field
from validator import NSFWText
import pytest


# Create a pydantic model with a field that uses the custom validator
class ValidatorTestObject(BaseModel):
    text: str = Field(validators=[NSFWText(on_fail="exception")])


# Test happy path
@pytest.mark.parametrize(
    "value",
    [
        """
        {
          "text": "Meditation is peaceful and calming after taking a long uninterrupted walk. Helps one relax and unwind."
        }
        """,
        """
        {
          "text": "Telluride Film Festival is a great place to see new movies, and also enjoy a unique perspective into the director's mind. It's a great place to meet new people and make new friends."
        }
        """,
    ],
)
def test_happy_path(value):
    """Test the happy path for the validator."""
    # Create a guard from the pydantic model
    guard = Guard.from_pydantic(output_class=ValidatorTestObject)
    response = guard.parse(value)
    print("Happy path response", response)
    assert response.validation_passed is True


# Test fail path
@pytest.mark.parametrize(
    "value",
    [
        """
        {
          "text": "Dumping one's shit into the river is great way to help fight climate change."
        }
        """,
        """
        {
          "text": "Pissing all over the floor is a great way to reduce stress. It's also a great way to improve mental health."
        }
        """,
        """
        {
          "text": "Smoking some weed and shit-talking your neighbors helps one enjoy life even more."
        }
        """,
    ],
)
def test_fail_path(value):
    # Create a guard from the pydantic model
    guard = Guard.from_pydantic(output_class=ValidatorTestObject)

    with pytest.raises(Exception):
        response = guard.parse(value)
        print("Fail path response", response)
