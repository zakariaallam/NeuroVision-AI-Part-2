import gdown

def load_data():
    fill_id = "1a4w4HfwvyzmdIF56TmWL5-z8of0ghE3m"
    Zip_Path = "data.zip"

    url = f"https://drive.google.com/uc?id={fill_id}"

    gdown.download(url,Zip_Path,quiet=False)