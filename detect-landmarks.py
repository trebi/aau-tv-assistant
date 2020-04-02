import json
from os import listdir
from os.path import isfile, join

def detect_landmarks(path):
    """Detects landmarks in the file."""
    from google.cloud import vision
    import io
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.landmark_detection(image=image)
    landmarks = response.landmark_annotations
    out_landmarks = {
        'file': path,
        'landmarks': []
    }

    for landmark in landmarks:
        out_landmark = {
            "description": landmark.description,
            'locations': []
        }

        for location in landmark.locations:
            lat_lng = location.lat_lng
            out_landmark['locations'].append({
                'latitude': lat_lng.latitude,
                'longitude': lat_lng.longitude
            })
        out_landmarks['landmarks'].append(out_landmark)

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    return out_landmarks

def main():
    path = "../images"
    with open("output/detected-landmarks.jsonl", "w") as output:
        for f in list(filter(lambda f: isfile(join(path, f)), listdir(path))):
            detected = detect_landmarks(join(path, f))
            output.write(json.dumps(detected) + "\n")

if __name__ == "__main__":
    main()
