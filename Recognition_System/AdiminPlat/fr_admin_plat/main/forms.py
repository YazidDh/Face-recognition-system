from django import forms
from .models import CustomUser



class addForm(forms.ModelForm):
   
    class Meta:
        model = CustomUser
        fields = ["username", "image"]