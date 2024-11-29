from pycocoevalcap.spice.spice import Spice

import json

predictions = [
    {
        "image_id": 1,
        "caption": "a person is riding a bike in the street"
    },
    {
        "image_id": 2,
        "caption": "a group of people sitting around a table"
    }
]

ground_truth = {
    1: [
        "a person is riding a bike down a city street",
        "a person is riding a bicycle down the street"
    ],
    2: [
        "a group of people sitting around a table in a restaurant",
        "a group of people sitting at a table"
    ]
}

predictions = {entry["image_id"]: entry["caption"] for entry in predictions}
gt = {key: value for key, value in ground_truth.items()}

spice = Spice()


spice_score = spice.compute_score(predictions, gt)

print("SPICE score:", spice_score)

