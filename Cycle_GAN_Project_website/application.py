from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from train import *
import argparse
import sys
import torch
application = Flask(__name__)

@application.route("/")
def hello():
    return render_template("hello.html")

@application.route("/apply")
def apply():
    return render_template("apply.html")

@application.route("/upload_done", methods=["POST"])
def upload_done():
    uploaded_files=request.files["file"]
    print(uploaded_files.filename)
    filename = secure_filename(uploaded_files.filename)
    url="static/img/testA/{}.jpg".format(filename)
    uploaded_files.save(url)
    test()
    url2="static/result/test/out3.png"
    return render_template("test.html",src=url,res=url2)

if __name__ == "__main__":
    application.run(host='0.0.0.0')
    print(torch.__version__)
