# from market.movile import mainthing
from sre_constants import SUCCESS
from market import app
from flask import render_template, redirect, url_for, flash, request, jsonify
from market.models import Item, User, datas
from market.forms import RegisterForm, LoginForm, Price
from market import db
from flask_login import login_user, logout_user, login_required
import pickle

@app.route('/')
@app.route('/home')
def home_page():
    return render_template('home.html')

@app.route('/market')
@login_required
def market_page():
    items = Item.query.all()
    return render_template('market.html', items=items)

@app.route('/register', methods=['GET', 'POST'])
def register_page():
    form = RegisterForm()
    if form.validate_on_submit():
        user_to_create = User(username=form.username.data,
                              email_address=form.email_address.data,
                              password=form.password1.data)
        db.session.add(user_to_create)
        db.session.commit()
        return redirect(url_for('market_page'))
    if form.errors != {}: #If there are not errors from the validations
        for err_msg in form.errors.values():
            flash(f'There was an error with creating a user: {err_msg}', category='danger')

    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    form = LoginForm()
    if form.validate_on_submit():
        attempted_user = User.query.filter_by(username=form.username.data).first()
        if attempted_user and attempted_user.check_password_correction(
                attempted_password=form.password.data
        ):
            login_user(attempted_user)
            flash(f'Success! You are logged in as: {attempted_user.username}', category='success')
            return redirect(url_for('home_page'))
        else:
            flash('Username and password are not match! Please try again', category='danger')

    return render_template('login.html', form=form)

@app.route('/logout')
def logout_page():
    logout_user()
    flash("You have been logged out!", category='info')
    return redirect(url_for("home_page"))

# @app.route('/price', methods=['GET','POST'])
# @login_required
# def predict():
#     form = Price()
#     # mv.reg.predict([[4.0,128.0,6.00,48,13.0,4000]])
#     if form.validate_on_submit():
#         Brand = form.Brand.data
#         Ratings= form.Ratings.data
#         Ram=form.Ram.data
#         Rom=form.Rom.data
#         Size=form.Size.data
#         pixel1=form.pixel1.data
#         pixel2=form.pixel2.data
#         battery=form.battery.data
#         print("Brand: ",Brand,"\n")
#         print("Ratings",Ratings,"\n")
#         print("Ram",Ram,"\n")
#         print("Rom",Rom,"\n")
#         print("Size",Size,"\n")
#         print("pixel1",pixel1,"\n")
#         print("pixel2",pixel2,"\n")
#         print("battery",battery,"\n")
#         res=mainthing(Ram,Rom,Size,pixel1,pixel2,battery)
#         flash(f' THE PREDICTED VALUE FOR YOUR DEVICE IS  Rs.{res}', category='success')
#         # return render_template('result.html',res=res)
#     return render_template('mobile.html',form=form)
    
@app.route('/getPrice', methods=['POST', 'GET'])
@login_required
def getPrice():
    if request.method== 'POST':
        res = request.values.to_dict()
        model = pickle.load(open('D:\Flask\FlaskMarket\market\model.sav', 'rb'))
        data = list(res.values())
        print(data)
        print("lllll",data[0])
        data_to_add = datas(ram=data[0],
                       rom=data[1],
                       Screen=data[2],
                       rearCam=data[3],
                       frontCam=data[4],
                       battery=data[5])
        db.session.add(data_to_add)
        db.session.commit()
        prediction = model.predict([data])
        # print(prediction)
        return jsonify({'price': prediction[0]})
    if request.method=='GET':
        return render_template('index.html')

