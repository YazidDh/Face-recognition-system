from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('add-user', views.add_user, name='add_post'),
    path('home', views.home, name='home'),
]
"""path('home', views.home, name='home'),
    path('add-post', views.create_post, name='add_post'),"""