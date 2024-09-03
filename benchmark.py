import time
from guardrails.hub import NSFWText
from guardrails import Guard
import matplotlib.pyplot as plt

test_dataset = [
    {"text": "Christopher Nolan's Tenet is a mind-bending action thriller that will keep you on the edge of your seat. The film is a must-watch for all Nolan fans.", "label": "SFW"},
    {"text": "Climate Change is real and we need to do something about it. Dumping one's shit into the river is a great way to help fight climate change.", "label": "NSFW"},
    {"text": "This is a clean and safe for work message.", "label": "SFW"},
    {"text": "This sentence contains a curse word: fuck.", "label": "NSFW"},
    {"text": "Let's discuss business strategies over lunch.", "label": "SFW"},
    {"text": "This text is full of inappropriate language like shit, damn, and hell.", "label": "NSFW"},
    {"text": "He's such a badass in that movie!", "label": "SFW"},  
    {"text": "I can't believe she said that! What a bitch!", "label": "NSFW"},  
    {"text": "The performance was killer!", "label": "SFW"},  
    {"text": "This is the shit!", "label": "SFW"},  
    {"text": "Stop being so damn annoying.", "label": "NSFW"},  
    {"text": "That party was lit!", "label": "SFW"},  
    {"text": "The new policy is crap.", "label": "NSFW"},  
    {"text": "I can't stand this damn weather.", "label": "NSFW"},  
    {"text": "He's a real pain in the ass sometimes.", "label": "NSFW"},  
    {"text": "You're such a genius!", "label": "SFW"},  
    {"text": "Screw that idea, it's not worth the effort.", "label": "NSFW"}, 
    {"text": "That movie was epic!", "label": "SFW"},  
    {"text": "What a joke, this plan is bullshit.", "label": "NSFW"}, 
    {"text": "I'm so fucking tired of this nonsense.", "label": "NSFW"}, 
]

guard = Guard().use(NSFWText, threshold=0.8, validation_method="sentence", on_fail="exception")

correct_sfw = 0
correct_nsfw = 0
incorrect_sfw_as_nsfw = 0
incorrect_nsfw_as_sfw = 0

misclassified_examples = []
latencies = []

for item in test_dataset:
    text = item["text"]
    label = item["label"]

    # Measure the time taken for each validation
    start_time = time.time()
    
    try:
        # Validate the text using the NSFWText validator
        guard.validate(text)
        if label == "SFW":
            correct_sfw += 1
        else:
            incorrect_nsfw_as_sfw += 1 
            misclassified_examples.append((text, "NSFW", "SFW"))
            print(f"Misclassified NSFW as SFW: {text}")
    except Exception:
        if label == "NSFW":
            correct_nsfw += 1
        else:
            incorrect_sfw_as_nsfw += 1  
            misclassified_examples.append((text, "SFW", "NSFW"))
            print(f"Misclassified SFW as NSFW: {text}")

    # Calculate and store latency
    latency = time.time() - start_time
    latencies.append(latency)

print("Finished processing all items")

# Create a figure with two subplots side by side
plt.figure(figsize=(14, 6))

# Subplot 1: Bar chart for classification results
plt.subplot(1, 2, 1)
categories = ['Correctly Classified SFW', 'Correctly Classified NSFW', 'SFW misclassified as NSFW', 'NSFW misclassified as SFW']
values = [correct_sfw, correct_nsfw, incorrect_sfw_as_nsfw, incorrect_nsfw_as_sfw]
colors = ['green', 'blue', 'red', 'orange']
plt.bar(categories, values, color=colors)
plt.title('NSFWText Validator Predictions')
plt.ylabel('Number of Examples')
plt.xticks(rotation=15, ha='right')  # Rotate x-axis labels to avoid overlap

# Subplot 2: Scatter plot for latency results
plt.subplot(1, 2, 2)
plt.scatter(range(len(latencies)), latencies, color='blue', label='Latency per Validation')
average_latency = sum(latencies) / len(latencies)
plt.axhline(y=average_latency, color='red', linestyle='--', label=f'Average Latency: {average_latency:.4f} sec')
plt.title('Validation Latencies')
plt.xlabel('Validation Instance')
plt.ylabel('Latency (seconds)')
plt.legend()

plt.tight_layout()
plt.show()

# Print misclassified examples
if misclassified_examples:
    print("\nMisclassified Examples:")
    for text, true_label, predicted_label in misclassified_examples:
        print(f"Text: {text}\nTrue Label: {true_label}, Predicted Label: {predicted_label}\n")
else:
    print("\nNo misclassified examples.")

print("Latency Information")
# Print latency information
print(f"\nAverage Latency per Validation: {average_latency:.4f} seconds")
print(f"Maximum Latency: {max(latencies):.4f} seconds")
print(f"Minimum Latency: {min(latencies):.4f} seconds")

print("End of script")
