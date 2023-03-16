import pdb
import json
import os
from flask import request
from flask import render_template
from flask import Flask
from chat import chat
from tools.vector import server_init
app = Flask(__name__)
server_init()


@app.route('/', methods=["GET", "POST"])
def index():
    return render_template("home.html")


@app.route('/search', methods=["GET", "POST"])
def search():
    text = request.args.get("q")
    print(text)
    output = chat(text)
    return json.dumps(output, ensure_ascii=False).encode("UTF-8")


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
