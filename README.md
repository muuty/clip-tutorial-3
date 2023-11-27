# clip-tutorial-3

# Description
This is a tutorial on how to use the open-ai clip with Fastapi.



# 1. Install
I recommend using a virtual environment. On the live session, I used pycharm to create a virtual environment.

After creating and activating a virtual environment, run the following command to install the required packages.
```bash
pip install -r requirements.txt
```


# 2. Usage
```bash
python main.py
```

# 3. How to use custom images
If you want to use custom images, put them in the `images` folder.

Then, run '1.extract_features.py' to extract the features of the images.

Finally, run 'main.py' to start the server.
