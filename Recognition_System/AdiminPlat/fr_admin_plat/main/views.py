from django.shortcuts import redirect, render
from .models import CustomUser
import os
from .forms import addForm

# Create your views here.
def home(request):
    users = CustomUser.objects.all()
    if request.method == "POST":
        user_id = request.POST.get("user-id")
        if user_id:
            user = CustomUser.objects.filter(id=user_id).first()
            if user:
                os.remove(fr'C:\Users\ASUS\OneDrive\Bureau\face(yazid)\AdiminPlat\fr_admin_plat\{user.image.url}')
                user.delete()
    return render(request, 'main/home.html',{"users": users})

def add_user(request):
    if request.method == 'POST':
        form = addForm(request.POST,  request.FILES)
        if form.is_valid():
           form.save()
           return redirect("/home")
    else:
        form = addForm()

    return render(request, 'main/add_user.html', {"form": form}) 
