from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import sys
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
    filename = secure_filename(uploaded_files.filename)
    real="static/img/{}.jpg".format(filename)
    uploaded_files.save(real)
    return render_template("test.html",url=real)

if __name__ == "__main__":
    application.run(host='0.0.0.0')
