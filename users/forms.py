from django import forms

from users.models import UserRegistrationModel, HeartDataModel


class UserRegistrationForm(forms.ModelForm):
    name = forms.CharField(widget=forms.TextInput(attrs={'pattern':'[a-zA-Z]+'}), required=True,max_length=100)
    loginid = forms.CharField(widget=forms.TextInput(attrs={'pattern':'[a-zA-Z]+'}), required=True,max_length=100)
    password = forms.CharField(widget=forms.PasswordInput(attrs={'pattern':'(?=.*\d)(?=.*[a-z])(?=.*[A-Z]).{8,}','title':'Must contain at least one number and one uppercase and lowercase letter, and at least 8 or more characters'}), required=True,max_length=100)
    mobile = forms.CharField(widget=forms.TextInput(attrs={'pattern':'[56789][0-9]{9}'}), required=True,max_length=100)
    email = forms.CharField(widget=forms.TextInput(attrs={'pattern':'[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$'}), required=True,max_length=100)
    locality = forms.CharField(widget=forms.TextInput(), required=True,max_length=100)
    address = forms.CharField(widget=forms.Textarea(attrs={'rows':4, 'cols': 22}), required=True,max_length=250)
    city = forms.CharField(widget=forms.TextInput(attrs={'class':'form-control' , 'autocomplete': 'off','pattern':'[A-Za-z ]+', 'title':'Enter Characters Only '}), required=True,max_length=100)
    state = forms.CharField(widget=forms.TextInput(attrs={'class':'form-control' , 'autocomplete': 'off','pattern':'[A-Za-z ]+', 'title':'Enter Characters Only '}), required=True,max_length=100)
    status = forms.CharField(widget=forms.HiddenInput(), initial='waiting' ,max_length=100)


    class Meta():
        model = UserRegistrationModel
        fields='__all__'



class HeartDataForm(forms.ModelForm):
    age = forms.IntegerField()
    sex = forms.IntegerField()
    cp = forms.IntegerField()
    trestbps = forms.IntegerField()
    chol = forms.IntegerField()
    fbs = forms.IntegerField()
    restecg = forms.IntegerField()
    thalach = forms.IntegerField()
    exang = forms.IntegerField()
    oldpeak = forms.FloatField()
    slope = forms.IntegerField()
    ca = forms.IntegerField()
    thal = forms.IntegerField()
    target = forms.IntegerField()



    class Meta():
        model = HeartDataModel
        fields = '__all__'