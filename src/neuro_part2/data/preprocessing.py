from sklearn.preprocessing import LabelEncoder
import cv2
from pathlib import Path
import zipfile


VALID_EXTANTION = ['.jpeg','.jpg','png','bmp']

def load_data(data_dir,image_size=(224,224)):

  data_dir = Path(data_dir)

  if data_dir.is_file() and data_dir.suffix.lower() == ".zip":
    extract_dir = data_dir.with_suffix("")
    with zipfile.ZipFile(data_dir,"r") as zip_ref:
      zip_ref.extractall(extract_dir)

    data_dir = extract_dir

    data_dir = data_dir / "Data"

  if not data_dir.is_dir():
    raise FileNotFoundError(
        f"dossier introvable : {data_dir}"
    )

  images = []
  labels = []
  class_names = sorted(
    p.name for p in data_dir.iterdir() if p.is_dir()
    )

  if not class_names:
      raise ValueError(
          "Aucun class trouvé"
      )

  encoded = LabelEncoder()
  encoded.fit(class_names)

  for class_name in class_names:
      class_dir = data_dir / class_name

      for path in class_dir.rglob("*"):

        if not path.is_file():
          continue

        if path.suffix.lower() not in VALID_EXTANTION :
          continue

        try:
            image = cv2.imread(str(path))

            image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)

            image = cv2.resize(image , image_size)

            images.append(image)
            labels.append(class_name)

        except Exception as e:

            print(f"image ignor {path} : {e}")

  x = images
  y = encoded.transform(labels)

  print("number des images valid : ",len(x))
  print("classes : ",class_names)
  # print("dimension : ",x.shape)

  return x , y , class_names