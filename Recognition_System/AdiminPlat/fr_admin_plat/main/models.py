from django.db import models

class CustomUser(models.Model):
    username = models.CharField(max_length=200)
    image = models.ImageField(upload_to='images/')