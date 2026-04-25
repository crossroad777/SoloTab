import os, glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_dir = r"D:\Music\nextchord-solotab\generated\fretnet_models\models\strumming_finetuned"
event_files = sorted(glob.glob(os.path.join(log_dir, "events.out.tfevents.*")))

if event_files:
    ea = EventAccumulator(event_files[-1])
    ea.Reload()
    tags = ea.Tags()["scalars"]
    if not tags:
        print("No scalar tags found. The model might not have completed the first validation loop yet.")
    else:
        print("Model Progress Summary:\n" + "-"*40)
        for tag in tags:
            events = ea.Scalars(tag)
            if events:
                print(f"{tag}: Iter {events[0].step} -> {events[0].value:.4f} | Iter {events[-1].step} -> {events[-1].value:.4f}")
else:
    print("No event files found.")
