import enum

class RunType(enum.Enum):
    TRAIN = 0
    TEST = 1


SETTINGS = {
    RunType.TRAIN: {
        "labels": ("./training/train-labels-idx1-ubyte", 2049),
        "images": ("./training/train-images-idx3-ubyte", 2051)
    },
    RunType.TEST: {
        "labels": ("./testing/t10k-labels.idx1.ubyte",2049),
        "images": ("./testing/t10k-images.idx3.ubyte", 2051)
    }
}

image = list[list[float]]

def read_image_chunks(run_type: RunType) -> list[image]:
    file_path, expected_magic = SETTINGS[run_type]["images"]
    with open(file_path, "rb") as f:
        chunk = f.read(4)
        if not chunk:
            raise RuntimeError("ERROR: Could not read file")
        if int.from_bytes(chunk, byteorder='big') != expected_magic:
            raise RuntimeError(f"ERROR: Invalid Magic. Got {int.from_bytes(chunk, byteorder='big')} but expected {expected_magic}")

        no_of_images = int.from_bytes(f.read(4), byteorder='big')
        no_of_rows = int.from_bytes(f.read(4), byteorder='big')
        no_of_cols = int.from_bytes(f.read(4), byteorder='big')
        print(no_of_images, no_of_cols, no_of_rows)
        images = []
        for _ in range(1):
            if len(images) % 10000 == 0:
                print(f"Read {len(images)} images so far")
            cur_image = []
            for _ in range(no_of_cols):
                cur_row = []
                for _ in range(no_of_rows):
                    cur_chunk = int.from_bytes(f.read(1), byteorder='big') / 255.0
                    if cur_chunk is None:
                        print(f"Error: Got a null!")
                        break
                    cur_row.append(cur_chunk) # read one pixel at a time
                cur_image.append(cur_row)
            images.append(cur_image)
    return images


def read_labels_chunks(run_type: RunType) -> list[int]:
    file_path, expected_magic = SETTINGS[run_type]["labels"]
    with open(file_path, "rb") as f:
        chunk = f.read(4)
        if not chunk:
            # eof
            return []
        magic = int.from_bytes(chunk, byteorder='big')
        if magic != expected_magic:
            print("ERROR: Invalid Magic. Possibly reading wrong file")
            raise RuntimeError("ERROR: Invalid Magic Provided")
        
        print(f"INFO: Magic success {magic}")
        number_of_items = int.from_bytes(f.read(4), byteorder='big')
        print(f"Number of items: {number_of_items}")
        labels: list[int] = []
        while True:
            chunk = f.read(1) 
            if not chunk:
                return labels
            labels.append(int.from_bytes(chunk, byteorder='big'))
        
    return labels

def display_image(pixels: image, expected_label):
    for row in pixels:
        print(row)
    print(expected_label)