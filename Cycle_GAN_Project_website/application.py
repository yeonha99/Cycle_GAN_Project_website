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
    
    ## Parser 생성하기
    parser = argparse.ArgumentParser(description="CycleGAN",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", default="test", choices=["train", "test"], type=str, dest="mode")
    parser.add_argument("--train_continue", default="on", choices=["on", "off"], type=str, dest="train_continue")
    parser.add_argument("--lr", default=2e-4, type=float, dest="lr")
    parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
    parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")
    parser.add_argument("--data_dir", default="./static/img", type=str, dest="data_dir")
    parser.add_argument("--ckpt_dir", default="./pth", type=str, dest="ckpt_dir")
    parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
    parser.add_argument("--result_dir", default="./static/result", type=str, dest="result_dir")

    parser.add_argument("--task", default="cyclegan", choices=['cyclegan'], type=str, dest="task")
    parser.add_argument('--opts', nargs='+', default=['direction', 0], dest='opts')

    parser.add_argument("--ny", default=256, type=int, dest="ny")
    parser.add_argument("--nx", default=256, type=int, dest="nx")
    parser.add_argument("--nch", default=3, type=int, dest="nch")
    parser.add_argument("--nker", default=64, type=int, dest="nker")

    parser.add_argument("--wgt_cycle", default=1e1, type=float, dest="wgt_cycle")
    parser.add_argument("--wgt_ident", default=5e-1, type=float, dest="wgt_ident")
    parser.add_argument("--norm", default='inorm', type=str, dest="norm")

    parser.add_argument("--network", default="CycleGAN", choices=['DCGAN', 'pix2pix', 'CycleGAN'], type=str,dest="network")
    parser.add_argument("--learning_type", default="plain", choices=["plain", "residual"], type=str,dest="learning_type")
    args = parser.parse_args()
    test(args)
    url2="static/result/test/out.png"
    return render_template("test.html",src=url,res=url2)

if __name__ == "__main__":
    application.run(host='0.0.0.0')
    print(torch.__version__)
