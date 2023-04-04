from numbers import Number
from flask_wtf import FlaskForm
from numpy import number
from wtforms import StringField, PasswordField, SubmitField, IntegerField, FloatField
from wtforms.validators import Length, EqualTo, Email, DataRequired, ValidationError
from market.models import User


class RegisterForm(FlaskForm):
    def validate_username(self, username_to_check):
        user = User.query.filter_by(username=username_to_check.data).first()
        if user:
            raise ValidationError('Username already exists! Please try a different username')

    def validate_email_address(self, email_address_to_check):
        email_address = User.query.filter_by(email_address=email_address_to_check.data).first()
        if email_address:
            raise ValidationError('Email Address already exists! Please try a different email address')

    username = StringField(label='User Name:', validators=[Length(min=2, max=30), DataRequired()])
    email_address = StringField(label='Email Address:', validators=[Email(), DataRequired()])
    password1 = PasswordField(label='Password:', validators=[Length(min=6), DataRequired()])
    password2 = PasswordField(label='Confirm Password:', validators=[EqualTo('password1'), DataRequired()])
    submit = SubmitField(label='Create Account')


class LoginForm(FlaskForm):
    username = StringField(label='User Name:', validators=[DataRequired()])
    password = PasswordField(label='Password:', validators=[DataRequired()])
    submit = SubmitField(label='Sign in')

class Price(FlaskForm):
    Brand = StringField(label='Brand name', validators=[DataRequired()])
    Ratings = FloatField(label='Device Rating', validators=[DataRequired()])
    Ram = IntegerField(label='RAM ', validators=[DataRequired()])
    Rom = IntegerField(label='ROM ', validators=[DataRequired()])
    Size = FloatField(label='Mobile Screen Size ', validators=[DataRequired()])
    pixel1 = IntegerField(label='Front Camera Pixels ', validators=[DataRequired()])
    pixel2 = IntegerField(label='Rear Camera Pixels ', validators=[DataRequired()])
    battery = IntegerField(label='Enter the battery', validators=[DataRequired()])

    submit = SubmitField(label='Submit')