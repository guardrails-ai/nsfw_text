# Import Guard and Validator
from NSFWText import NSFWText
from guardrails import Guard

# Setup Guard with the validator
guard = Guard().use(
    NSFWText, threshold=0.8, validation_method="sentence", on_fail="exception", use_local=False, validation_endpoint="http://127.0.01:8000/validate"
)

# Test passing response
print(guard.validate(
    "Christopher Nolan's Tenet is a shit action thriller that will keep you on the edge of your seat. The film is a must-watch for all Nolan fans."
))

# try:
#     # Test failing response
#     guard.validate(
#         "Climate Change is real and we need to do something about it. Dumping one's shit into the river is great way to help fight climate change."
#     )
# except Exception as e:
#     print(e)


# , use_local=False, validation_endpoint="http://127.0.01:8000/validate"
