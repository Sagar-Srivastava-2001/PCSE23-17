import PIL
from flask import Flask, render_template, request

import Generate_caption
import Generate_caption_1
from Detect_Head_Count import detect
app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/',methods=["POST"])
def prediction():
    if request.method=="POST":
        f = request.files['userfile']
        path = 'static/{}'.format(f.filename)
        f.save(path)
        caption = Generate_caption.caption_image(path)
        caption1 = Generate_caption_1.caption_image(path)
        total_head = 0
        try:
            total_head = detect(path)[1]['person']
        except Exception as e:
            print(e)
            total_head = 0
        result = {'image':path,'cap':caption,'caption1':caption1,'total_head':total_head}
        
    return render_template('index.html',your_caption = result)

if __name__== '__main__':
    app.run(debug=False)

