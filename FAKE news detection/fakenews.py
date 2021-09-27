from flask import Flask, render_template, request, session, logging, flash, url_for, redirect, jsonify, Response
import json
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
from flask_mail import Mail
import os

#model
from flask import Flask, abort, jsonify, request, render_template
from sklearn.externals import joblib
from feature import *
import json

with open('config.json', 'r') as c:
    params = json.load(c)["params"]
# Define a flask app
local_server = True
app = Flask(__name__, template_folder='template')
app.secret_key = 'super-secret-key'

pipeline = joblib.load('./pipeline.sav')

app.config['MAIL_SERVER'] = 'smtp.googlemail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = params['gmail_user']
app.config['MAIL_PASSWORD'] = params['gmail_password']
mail = Mail(app)


if(local_server):
    app.config['SQLALCHEMY_DATABASE_URI'] = params['local_uri']
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = params['prod_uri']

db = SQLAlchemy(app)

class Contact(db.Model):
    '''
    sno, name phone_num, msg, date, email
    '''
    sno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(20), nullable=False)
    phone_num = db.Column(db.String(12), nullable=False)
    message = db.Column(db.String(120), nullable=False)
    date = db.Column(db.String(12), nullable=True)

class Register(db.Model):
    '''
    sno, name phone_num, msg, date, email
    '''
    rno = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(20), nullable=False)
    password = db.Column(db.String(12), nullable=False)
    password2 = db.Column(db.String(120), nullable=False)


@app.route("/")
def home():
    return render_template('index.html', params=params)

@app.route("/about")
def about():
    return render_template('about.html', params=params)

@app.route("/register", methods=['GET', 'POST'])
def register():
    if(request.method == 'POST'):
        # import pdb;pdb.set_trace();
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        password2 = request.form.get('password2')
        error=""
        avilable_email= Register.query.filter_by(email=email).first()
        if avilable_email:
            error="email is already exists"
        else:
            if(password==password2):
                entry = Register(name=name,email=email,password=password, password2=password2)
                db.session.add(entry)
                db.session.commit()
                return redirect(url_for('login'))
            else:
                flash("plz enter right password")
        return render_template('register.html',params=params, error=error)
    return render_template('register.html', params=params)

    
@app.route("/contact", methods=['GET', 'POST'])
def contact():
    if(request.method == 'POST'):
        name = request.form.get('name')
        email = request.form.get('email')
        phone = request.form.get('contact')
        message = request.form.get('message')
        entry = Contact(name=name, phone_num=phone,
                        message=message, email=email, date=datetime.now())
        db.session.add(entry)
        db.session.commit()
    return render_template('contact.html', params=params)



@app.route("/login",methods=['GET','POST'])
def login():
    if (request.method== "GET"):
        if('email' in session and session['email']):
            return render_template('fakenews.html',params=params)
        else:
            return render_template("login.html", params=params)
    if (request.method== "POST"):
        email = request.form["email"]
        password = request.form["password"]
        
        login = Register.query.filter_by(email=email, password=password).first()
        if login is not None:
            session['email']=email
            return render_template('fakenews.html',params=params)
        else:
            flash("plz enter right password")
    return render_template('login.html',params=params)


# @app.route("/", methods=['GET','POST'])
# def fakenews():
#     return render_template('fakenews.html', params=params)
@app.route('/')
def fakenews():
    return render_template('fakenews.html')

@app.route('/api',methods=['POST'])
def get_delay():
    import pdb;pdb.set_trace();
    result=request.form
    query_title = result['title']
    query_author = result['author']
    query_text = result['maintext']
    print(query_text)
    query = get_all_query(query_title, query_author, query_text)
    user_input = {'query':query}
    pred = pipeline.predict(query)
    print(pred)#    
    dic = {1:'real',0:'fake'}
    return f'<html><body><h1>{dic[pred[0]]}</h1> <form action="/"> <button type="submit">back </button> </form></body></html>'


@app.route("/logout", methods=['GET','POST'])
def logout():
    session.pop('email')
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
